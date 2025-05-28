from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import einops
from einops import repeat

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from ..embeddings import PatchEmbed, PooledEmbed, TimestepEmbed, EmbedND, OutEmbed
from ..attention import HiDreamAttention, FeedForwardSwiGLU
from ..attention_processor import HiDreamAttnProcessor_flashattn
from diffusers.models.attention_processor import AttentionProcessor
from ..moe import MOEFeedForwardSwiGLU

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class TextProjection(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=hidden_size, bias=False)

    def forward(self, caption):
        hidden_states = self.linear(caption)
        return hidden_states
    
class BlockType:
    TransformerBlock = 1
    SingleTransformerBlock = 2

@maybe_allow_in_graph
class HiDreamImageSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        _force_inference_output: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # 1. Attention
        self.norm1_i = nn.LayerNorm(dim, eps = 1e-06, elementwise_affine = False)
        self.attn1 = HiDreamAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            processor = HiDreamAttnProcessor_flashattn(),
            single = True,
        )

        # 3. Feed-forward
        self.norm3_i = nn.LayerNorm(dim, eps = 1e-06, elementwise_affine = False)
        if num_routed_experts > 0:
            self.ff_i = MOEFeedForwardSwiGLU(
                dim = dim, 
                hidden_dim = 4 * dim,
                num_routed_experts = num_routed_experts,
                num_activated_experts = num_activated_experts,
                _force_inference_output=_force_inference_output,
            )
        else:
            self.ff_i = FeedForwardSwiGLU(dim = dim, hidden_dim = 4 * dim)
    
    def forward(
        self,
        image_tokens: torch.FloatTensor,
        cond_tokens: Optional[torch.FloatTensor] = None,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        adaln_input: Optional[torch.FloatTensor] = None,
        cond_adaln_input: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        use_cond = cond_tokens is not None
        wtype = image_tokens.dtype
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i = \
            self.adaLN_modulation(adaln_input)[:,None].chunk(6, dim=-1)
        
        if use_cond:
            cond_shift_msa_i, cond_scale_msa_i, cond_gate_msa_i, cond_shift_mlp_i, cond_scale_mlp_i, cond_gate_mlp_i = \
            self.adaLN_modulation(cond_adaln_input)[:,None].chunk(6, dim=-1)
        
        # 1. MM-Attention
        norm_image_tokens = self.norm1_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_msa_i) + shift_msa_i
        if use_cond:
            norm_cond_tokens = self.norm1_i(cond_tokens).to(dtype=wtype)
            norm_cond_tokens = norm_cond_tokens * (1 + cond_scale_msa_i) + cond_shift_msa_i
            norm_image_tokens = torch.concat([norm_image_tokens, norm_cond_tokens], dim=-2)

        attn_output_i = self.attn1(
            norm_image_tokens,
            image_tokens_masks,
            use_cond = use_cond,
            rope = rope,
        )

        if use_cond:
            attn_output_i, cond_attn_output = attn_output_i
        else:
            cond_attn_output = None

        image_tokens = gate_msa_i * attn_output_i + image_tokens
        if use_cond:
            cond_tokens = cond_gate_msa_i * cond_attn_output + cond_tokens
        
        # 2. Feed-forward
        norm_image_tokens = self.norm3_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_mlp_i) + shift_mlp_i
        ff_output_i = gate_mlp_i * self.ff_i(norm_image_tokens.to(dtype=wtype))
        image_tokens = ff_output_i + image_tokens
        if use_cond:
            norm_cond_tokens = self.norm3_i(cond_tokens).to(dtype=wtype)
            norm_cond_tokens = norm_cond_tokens * (1 + cond_scale_mlp_i) + cond_shift_mlp_i
            cond_ff_output = cond_gate_mlp_i * self.ff_i(norm_cond_tokens)
            cond_tokens = cond_ff_output + cond_tokens

        return image_tokens, cond_tokens

@maybe_allow_in_graph
class HiDreamImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
         _force_inference_output: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 12 * dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # 1. Attention
        self.norm1_i = nn.LayerNorm(dim, eps = 1e-06, elementwise_affine = False)
        self.norm1_t = nn.LayerNorm(dim, eps = 1e-06, elementwise_affine = False)
        self.attn1 = HiDreamAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            processor = HiDreamAttnProcessor_flashattn(),
            single = False
        )

        # 3. Feed-forward
        self.norm3_i = nn.LayerNorm(dim, eps = 1e-06, elementwise_affine = False)
        if num_routed_experts > 0:
            self.ff_i = MOEFeedForwardSwiGLU(
                dim = dim, 
                hidden_dim = 4 * dim,
                num_routed_experts = num_routed_experts,
                num_activated_experts = num_activated_experts,
                _force_inference_output=_force_inference_output,
            )
        else:
            self.ff_i = FeedForwardSwiGLU(dim = dim, hidden_dim = 4 * dim)
        self.norm3_t = nn.LayerNorm(dim, eps = 1e-06, elementwise_affine = False)
        self.ff_t = FeedForwardSwiGLU(dim = dim, hidden_dim = 4 * dim)
    
    def forward(
        self,
        image_tokens: torch.FloatTensor,
        cond_tokens: Optional[torch.FloatTensor] = None,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        adaln_input: Optional[torch.FloatTensor] = None,
        cond_adaln_input: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        use_cond = cond_tokens is not None
        wtype = image_tokens.dtype
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i, \
        shift_msa_t, scale_msa_t, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = \
            self.adaLN_modulation(adaln_input)[:,None].chunk(12, dim=-1)
        
        if use_cond:
            cond_shift_msa_i, cond_scale_msa_i, cond_gate_msa_i, cond_shift_mlp_i, cond_scale_mlp_i, cond_gate_mlp_i,\
            _, _, _, _, _, _ = self.adaLN_modulation(cond_adaln_input)[:, None].chunk(12, dim=-1)
        
        # 1. MM-Attention
        norm_image_tokens = self.norm1_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_msa_i) + shift_msa_i
        norm_text_tokens = self.norm1_t(text_tokens).to(dtype=wtype)
        norm_text_tokens = norm_text_tokens * (1 + scale_msa_t) + shift_msa_t

        if use_cond:
            norm_cond_tokens = self.norm1_i(cond_tokens).to(dtype=wtype)
            norm_cond_tokens = norm_cond_tokens * (1 + cond_scale_msa_i) + cond_shift_msa_i
            norm_image_tokens = torch.concat([norm_image_tokens, norm_cond_tokens], dim=-2)

        attn_output = self.attn1(
            norm_image_tokens,
            image_tokens_masks,
            norm_text_tokens,
            rope = rope,
        )

        attn_output_i, attn_output_t = attn_output[:2]
        cond_attn_output = attn_output[2] if use_cond else None

        image_tokens = gate_msa_i * attn_output_i + image_tokens
        text_tokens = gate_msa_t * attn_output_t + text_tokens
        if use_cond:
            cond_tokens = cond_gate_msa_i * cond_attn_output + cond_tokens
        
        # 2. Feed-forward
        norm_image_tokens = self.norm3_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_mlp_i) + shift_mlp_i
        norm_text_tokens = self.norm3_t(text_tokens).to(dtype=wtype)
        norm_text_tokens = norm_text_tokens * (1 + scale_mlp_t) + shift_mlp_t
        if use_cond:
            norm_cond_tokens = self.norm3_i(cond_tokens).to(dtype=wtype)
            norm_cond_tokens = norm_cond_tokens * (1 + cond_scale_mlp_i) + cond_shift_mlp_i

        ff_output_i = gate_mlp_i * self.ff_i(norm_image_tokens)
        ff_output_t = gate_mlp_t * self.ff_t(norm_text_tokens)
        image_tokens = ff_output_i + image_tokens
        text_tokens = ff_output_t + text_tokens
        if use_cond:
            cond_ff_output = cond_gate_mlp_i * self.ff_i(norm_cond_tokens)
            cond_tokens = cond_ff_output + cond_tokens

        return image_tokens, text_tokens, cond_tokens
    
@maybe_allow_in_graph
class HiDreamImageBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        block_type: BlockType = BlockType.TransformerBlock,
         _force_inference_output=False
    ):
        super().__init__()
        block_classes = {
            BlockType.TransformerBlock: HiDreamImageTransformerBlock,
            BlockType.SingleTransformerBlock: HiDreamImageSingleTransformerBlock,
        }
        self.block = block_classes[block_type](
            dim,
            num_attention_heads,
            attention_head_dim,
            num_routed_experts,
            num_activated_experts,
            _force_inference_output
        )
    
    def forward(
        self,
        image_tokens: torch.FloatTensor,
        cond_tokens: Optional[torch.FloatTensor] = None,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        adaln_input: torch.FloatTensor = None,
        cond_adaln_input: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        return self.block(
            image_tokens,
            cond_tokens,
            image_tokens_masks,
            text_tokens,
            adaln_input,
            cond_adaln_input,
            rope,
        )

class HiDreamImageTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin
):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["HiDreamImageBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Optional[int] = None,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 16,
        num_single_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 20,
        caption_channels: List[int] = None,
        text_emb_dim: int = 2048,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        axes_dims_rope: Tuple[int, int] = (32, 32),
        max_resolution: Tuple[int, int] = (128, 128),
        llama_layers: List[int] = None, 
        force_inference_output: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.llama_layers = llama_layers

        self.t_embedder = TimestepEmbed(self.inner_dim)
        self.p_embedder = PooledEmbed(text_emb_dim, self.inner_dim)
        self.x_embedder = PatchEmbed(
            patch_size = patch_size,
            in_channels = in_channels,
            out_channels = self.inner_dim,
        )
        self.pe_embedder = EmbedND(theta=10000, axes_dim=axes_dims_rope)

        self.double_stream_blocks = nn.ModuleList(
            [
                HiDreamImageBlock(
                    dim = self.inner_dim,
                    num_attention_heads = self.config.num_attention_heads,
                    attention_head_dim = self.config.attention_head_dim,
                    num_routed_experts = num_routed_experts,
                    num_activated_experts = num_activated_experts,
                    block_type = BlockType.TransformerBlock,
                    _force_inference_output=force_inference_output,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_stream_blocks = nn.ModuleList(
            [
                HiDreamImageBlock(
                    dim = self.inner_dim,
                    num_attention_heads = self.config.num_attention_heads,
                    attention_head_dim = self.config.attention_head_dim,
                    num_routed_experts = num_routed_experts,
                    num_activated_experts = num_activated_experts,
                    block_type = BlockType.SingleTransformerBlock,
                    _force_inference_output=force_inference_output,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.final_layer = OutEmbed(self.inner_dim, patch_size, self.out_channels)

        caption_channels = [caption_channels[1], ] * (num_layers + num_single_layers) + [caption_channels[0], ]
        caption_projection = []
        for caption_channel in caption_channels:
            caption_projection.append(TextProjection(in_features = caption_channel, hidden_size = self.inner_dim))
        self.caption_projection = nn.ModuleList(caption_projection)
        self.max_seq = max_resolution[0] * max_resolution[1] // (patch_size * patch_size)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def expand_timesteps(self, timesteps, batch_size, device):
        if not torch.is_tensor(timesteps):
            is_mps = device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(batch_size)
        return timesteps

    def unpatchify(self, x: torch.Tensor, img_sizes: List[Tuple[int, int]], is_training: bool) -> List[torch.Tensor]:   
        is_training = False         
        if is_training:
            # x = einops.rearrange(x, 'B S (p1 p2 C) -> B C S (p1 p2)', p1=self.config.patch_size, p2=self.config.patch_size)
            B, S, F = x.shape
            C = F // (self.config.patch_size * self.config.patch_size)
            x = x.reshape(B, S, self.config.patch_size, self.config.patch_size, C).permute(0, 4, 1, 2, 3).reshape(B, C, S, self.config.patch_size * self.config.patch_size)
            print(B, S, F)
            print(x.shape)
        else:
            x_arr = []
            for i, img_size in enumerate(img_sizes):
                pH, pW = img_size
                x_arr.append(
                    einops.rearrange(x[i, :pH*pW].reshape(1, pH, pW, -1), 'B H W (p1 p2 C) -> B C (H p1) (W p2)', 
                        p1=self.config.patch_size, p2=self.config.patch_size)
                )
            x = torch.cat(x_arr, dim=0)
        return x

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def patchify(self, x, max_seq, img_sizes=None):
        pz2 = self.config.patch_size * self.config.patch_size
        if isinstance(x, torch.Tensor):
            B, C = x.shape[0], x.shape[1]
            device = x.device
            dtype = x.dtype
        else:
            B, C = len(x), x[0].shape[0]
            device = x[0].device
            dtype = x[0].dtype
        x_masks = torch.zeros((B, max_seq), dtype=dtype, device=device)

        if img_sizes is not None:
            for i, img_size in enumerate(img_sizes):
                x_masks[i, 0:img_size[0] * img_size[1]] = 1
            x = einops.rearrange(x, 'B C S p -> B S (p C)', p=pz2)
        elif isinstance(x, torch.Tensor):
            pH, pW = x.shape[-2] // self.config.patch_size, x.shape[-1] // self.config.patch_size
            x = einops.rearrange(x, 'B C (H p1) (W p2) -> B (H W) (p1 p2 C)', p1=self.config.patch_size, p2=self.config.patch_size)
            img_sizes = [[pH, pW]] * B
            x_masks = None
        else:
            raise NotImplementedError
        return x, x_masks, img_sizes

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_hidden_states: torch.Tensor = None,
        cond_img_ids: torch.Tensor = None,
        timesteps: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        pooled_embeds: torch.Tensor = None,
        img_sizes: Optional[List[Tuple[int, int]]] = None,
        img_ids: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        
        # Check if conditioning is enabled
        use_condition = cond_hidden_states is not None
            
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # spatial forward
        batch_size = hidden_states.shape[0]
        hidden_states_type = hidden_states.dtype

        # 0. time
        timesteps = self.expand_timesteps(timesteps, batch_size, hidden_states.device)
        cond_timesteps = torch.ones_like(timesteps) * 0
        timesteps = self.t_embedder(timesteps, hidden_states_type)
        p_embedder = self.p_embedder(pooled_embeds)
        adaln_input = timesteps + p_embedder
        cond_timesteps = self.t_embedder(cond_timesteps, hidden_states_type)
        cond_adaln_input = timesteps + p_embedder

        hidden_states, image_tokens_masks, img_sizes = self.patchify(hidden_states, self.max_seq, img_sizes)

        if image_tokens_masks is None:
            pH, pW = img_sizes[0]
            img_ids = torch.zeros(pH, pW, 3, device=hidden_states.device)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH, device=hidden_states.device)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW, device=hidden_states.device)[None, :]
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)

        hidden_states = self.x_embedder(hidden_states)
        
        if use_condition:
            latents_to_concat = []
            for i in range(cond_hidden_states.shape[0]):
                cond_latent, cond_tokens_masks, cond_size = self.patchify(cond_hidden_states[i], self.max_seq)
                latents_to_concat.append(cond_latent)
            cond_hidden_states = torch.concat(latents_to_concat, dim=-2)
            cond_hidden_states = self.x_embedder(cond_hidden_states)
            img_ids = torch.concat([img_ids, cond_img_ids], dim=-2)
        
        img_ids = img_ids.to(dtype=hidden_states.dtype)

        T5_encoder_hidden_states = encoder_hidden_states[0]
        encoder_hidden_states = encoder_hidden_states[-1]
        encoder_hidden_states = [encoder_hidden_states[k] for k in self.llama_layers]

        if self.caption_projection is not None:
            new_encoder_hidden_states = []
            for i, enc_hidden_state in enumerate(encoder_hidden_states):
                enc_hidden_state = self.caption_projection[i](enc_hidden_state)
                enc_hidden_state = enc_hidden_state.view(batch_size, -1, hidden_states.shape[-1])
                new_encoder_hidden_states.append(enc_hidden_state)
            encoder_hidden_states = new_encoder_hidden_states
            T5_encoder_hidden_states = self.caption_projection[-1](T5_encoder_hidden_states)
            T5_encoder_hidden_states = T5_encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
            encoder_hidden_states.append(T5_encoder_hidden_states)

        txt_ids = torch.zeros(
            batch_size, 
            encoder_hidden_states[-1].shape[1] + encoder_hidden_states[-2].shape[1] + encoder_hidden_states[0].shape[1], 
            3, 
            device=img_ids.device, dtype=img_ids.dtype
        )
        ids = torch.cat((img_ids, txt_ids), dim=1)
        rope = self.pe_embedder(ids).to(dtype=hidden_states.dtype)

        logger.debug(f"hidden_states.shape: {hidden_states.shape}")
        logger.debug(f"cond_hidden_states.shape: {cond_hidden_states.shape if cond_hidden_states is not None else None}")
        logger.debug(f"img_ids.shape: {img_ids.shape}")
        logger.debug(f"txt_ids.shape: {txt_ids.shape}")
        logger.debug(f"rope.shape: {rope.shape}")

        # 2. Blocks
        block_id = 0
        initial_encoder_hidden_states = torch.cat([encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1)
        initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]
        for bid, block in enumerate(self.double_stream_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            cur_encoder_hidden_states = torch.cat([initial_encoder_hidden_states, cur_llama31_encoder_hidden_states], dim=1)
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                if use_condition:
                    hidden_states, initial_encoder_hidden_states, cond_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,             # image_tokens
                        cond_hidden_states,        # cond_tokens
                        image_tokens_masks,        # image_tokens_masks
                        cur_encoder_hidden_states, # text_tokens
                        adaln_input,               # adaln_input
                        cond_adaln_input,          # cond_adaln_input
                        rope,                      # rope
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states, initial_encoder_hidden_states, cond_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,             # image_tokens
                        image_tokens_masks,        # image_tokens_masks
                        cur_encoder_hidden_states, # text_tokens
                        adaln_input,               # adaln_input
                        rope,                      # rope
                        **ckpt_kwargs,
                    )
            else:
                hidden_states, initial_encoder_hidden_states, cond_hidden_states = block(
                    image_tokens = hidden_states,
                    image_tokens_masks = image_tokens_masks,
                    text_tokens = cur_encoder_hidden_states,
                    adaln_input = adaln_input,
                    rope = rope,
                    cond_tokens=cond_hidden_states if use_condition else None,
                    cond_adaln_input=cond_adaln_input if use_condition else None,
                )
            initial_encoder_hidden_states = initial_encoder_hidden_states[:, :initial_encoder_hidden_states_seq_len]
            block_id += 1

        image_tokens_seq_len = hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
        hidden_states_seq_len = hidden_states.shape[1]
        if image_tokens_masks is not None:
            encoder_attention_mask_ones = torch.ones(
                (batch_size, initial_encoder_hidden_states.shape[1] + cur_llama31_encoder_hidden_states.shape[1]),
                device=image_tokens_masks.device, dtype=image_tokens_masks.dtype
            )
            image_tokens_masks = torch.cat([image_tokens_masks, encoder_attention_mask_ones], dim=1)

        for bid, block in enumerate(self.single_stream_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            hidden_states = torch.cat([hidden_states, cur_llama31_encoder_hidden_states], dim=1)
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward
                
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                if use_condition:
                    hidden_states, cond_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        cond_hidden_states,
                        image_tokens_masks,
                        None,
                        adaln_input,
                        cond_adaln_input,
                        rope,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states, cond_hidden_stat = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        image_tokens_masks,
                        None,
                        adaln_input,
                        rope,
                        **ckpt_kwargs,
                    )
            else:
                hidden_states, cond_hidden_stat = block(
                    image_tokens = hidden_states,
                    image_tokens_masks = image_tokens_masks,
                    text_tokens = None,
                    adaln_input = adaln_input,
                    cond_tokens=cond_hidden_states if use_condition else None,
                    cond_adaln_input=cond_adaln_input if use_condition else None,
                    rope = rope,
                )
            hidden_states = hidden_states[:, :hidden_states_seq_len]
            block_id += 1
        
        hidden_states = hidden_states[:, :image_tokens_seq_len, ...]
        output = self.final_layer(hidden_states, adaln_input)
        output = self.unpatchify(output, img_sizes, self.training)
        if image_tokens_masks is not None:
            image_tokens_masks = image_tokens_masks[:, :image_tokens_seq_len]

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, image_tokens_masks)
        return Transformer2DModelOutput(sample=output, mask=image_tokens_masks)
        
'''
DIFFUSION_TOOLKIT_PATH=. NUM_OVERFIT_SAMPLES=50 accelerate launch --config_file hidream/control/configs/accelerate_ds.yaml hidream/control/train_hidream_s2c.py --config_path hidream/control/configs/hidream_s2c_defaults.yaml
'''