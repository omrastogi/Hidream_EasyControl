import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import math
import einops
from einops import repeat
import torch
from torchvision.transforms.functional import pad
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerFast
)

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from .pipeline_output import HiDreamImagePipelineOutput
from ...models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel
from ...schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
        encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class HiDreamImagePipeline(DiffusionPipeline, FromSingleFileMixin):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->text_encoder_4->image_encoder->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5Tokenizer,
        text_encoder_4: LlamaForCausalLM,
        tokenizer_4: PreTrainedTokenizerFast,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            text_encoder_4=text_encoder_4,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            tokenizer_4=tokenizer_4,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        # HiDreamImage latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.default_sample_size = 128
        self.tokenizer_4.pad_token = self.tokenizer_4.eos_token

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_3.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, self.tokenizer_3.model_max_length),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, min(max_sequence_length, self.tokenizer_3.model_max_length) - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {min(max_sequence_length, self.tokenizer_3.model_max_length)} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device), attention_mask=attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        return prompt_embeds
    
    def _get_clip_prompt_embeds(
        self,
        tokenizer,
        text_encoder,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, 218),
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, 218 - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {218} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds
    
    def _get_llama3_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_4.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_4(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, self.tokenizer_4.model_max_length),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer_4(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_4.batch_decode(untruncated_ids[:, min(max_sequence_length, self.tokenizer_4.model_max_length) - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {min(max_sequence_length, self.tokenizer_4.model_max_length)} tokens: {removed_text}"
            )

        outputs = self.text_encoder_4(
            text_input_ids.to(device), 
            attention_mask=attention_mask.to(device), 
            output_hidden_states=True,
            output_attentions=True
        )

        prompt_embeds = outputs.hidden_states[1:]
        prompt_embeds = torch.stack(prompt_embeds, dim=0)
        _, _, seq_len, dim = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, 1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(-1, batch_size * num_images_per_prompt, seq_len, dim)
        return prompt_embeds
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        prompt_4: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        negative_prompt_4: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 128,
        lora_scale: Optional[float] = None,
    ):

        # Lora_scale might be needed to worked upon
        """
        Code from Flux Pipeline
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(
            prompt = prompt,
            prompt_2 = prompt_2,
            prompt_3 = prompt_3,
            prompt_4 = prompt_4,
            device = device,
            dtype = dtype,
            num_images_per_prompt = num_images_per_prompt,
            prompt_embeds = prompt_embeds,
            pooled_prompt_embeds = pooled_prompt_embeds,
            max_sequence_length = max_sequence_length,
        )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt
            negative_prompt_4 = negative_prompt_4 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )
            negative_prompt_4 = (
                batch_size * [negative_prompt_4] if isinstance(negative_prompt_4, str) else negative_prompt_4
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            
            negative_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt(
                prompt = negative_prompt,
                prompt_2 = negative_prompt_2,
                prompt_3 = negative_prompt_3,
                prompt_4 = negative_prompt_4,
                device = device,
                dtype = dtype,
                num_images_per_prompt = num_images_per_prompt,
                prompt_embeds = negative_prompt_embeds,
                pooled_prompt_embeds = negative_pooled_prompt_embeds,
                max_sequence_length = max_sequence_length,
            )
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        prompt_4: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 128,
    ):
        device = device or self._execution_device
        
        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_4 = prompt_4 or prompt
            prompt_4 = [prompt_4] if isinstance(prompt_4, str) else prompt_4

            pooled_prompt_embeds_1 = self._get_clip_prompt_embeds(
                self.tokenizer,
                self.text_encoder,
                prompt = prompt,
                num_images_per_prompt = num_images_per_prompt,
                max_sequence_length = max_sequence_length,
                device = device,
                dtype = dtype,
            )

            pooled_prompt_embeds_2 = self._get_clip_prompt_embeds(
                self.tokenizer_2,
                self.text_encoder_2,
                prompt = prompt_2,
                num_images_per_prompt = num_images_per_prompt,
                max_sequence_length = max_sequence_length,
                device = device,
                dtype = dtype,
            )

            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)

            t5_prompt_embeds = self._get_t5_prompt_embeds(
                prompt = prompt_3,
                num_images_per_prompt = num_images_per_prompt,
                max_sequence_length = max_sequence_length,
                device = device,
                dtype = dtype
            )
            llama3_prompt_embeds = self._get_llama3_prompt_embeds(
                prompt = prompt_4,
                num_images_per_prompt = num_images_per_prompt,
                max_sequence_length = max_sequence_length,
                device = device,
                dtype = dtype
            )
            prompt_embeds = [t5_prompt_embeds, llama3_prompt_embeds]

        return prompt_embeds, pooled_prompt_embeds

    # Copied and edited from https://github.com/huggingface/diffusers/blob/4267d8f4eb98449d9d29ffbb087d9bdd7690dbab/src/diffusers/pipelines/hidream_image/pipeline_hidream_image.py#L307
    def encode_all_prompt(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_4: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        negative_prompt_4: Optional[Union[str, List[str]]] = None,
        prompt_embeds_t5: Optional[List[torch.FloatTensor]] = None,
        prompt_embeds_llama3: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds_t5: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds_llama3: Optional[List[torch.FloatTensor]] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 128,
        lora_scale: Optional[float] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = pooled_prompt_embeds.shape[0]

        device = device or self._execution_device

        if pooled_prompt_embeds is None:
            pooled_prompt_embeds_1 = self._get_clip_prompt_embeds(
                self.tokenizer, 
                self.text_encoder, 
                prompt = prompt, 
                num_images_per_prompt = num_images_per_prompt,
                max_sequence_length = max_sequence_length,
                device = device,
                dtype = dtype,
            )

        if do_classifier_free_guidance and negative_pooled_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if len(negative_prompt) > 1 and len(negative_prompt) != batch_size:
                raise ValueError(f"negative_prompt must be of length 1 or {batch_size}")

            negative_pooled_prompt_embeds_1 = self._get_clip_prompt_embeds(
                self.tokenizer, 
                self.text_encoder, 
                prompt = negative_prompt, 
                num_images_per_prompt = num_images_per_prompt,
                max_sequence_length = max_sequence_length,
                device = device,
                dtype = dtype,
            )

            if negative_pooled_prompt_embeds_1.shape[0] == 1 and batch_size > 1:
                negative_pooled_prompt_embeds_1 = negative_pooled_prompt_embeds_1.repeat(batch_size, 1)

        if pooled_prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            if len(prompt_2) > 1 and len(prompt_2) != batch_size:
                raise ValueError(f"prompt_2 must be of length 1 or {batch_size}")

            pooled_prompt_embeds_2 = self._get_clip_prompt_embeds(
                self.tokenizer_2, 
                self.text_encoder_2, 
                prompt_2,
                num_images_per_prompt = num_images_per_prompt,
                max_sequence_length = max_sequence_length,
                device = device,
                dtype = dtype,  
            )

            if pooled_prompt_embeds_2.shape[0] == 1 and batch_size > 1:
                pooled_prompt_embeds_2 = pooled_prompt_embeds_2.repeat(batch_size, 1)

        if do_classifier_free_guidance and negative_pooled_prompt_embeds is None:
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_2 = [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2

            if len(negative_prompt_2) > 1 and len(negative_prompt_2) != batch_size:
                raise ValueError(f"negative_prompt_2 must be of length 1 or {batch_size}")

            negative_pooled_prompt_embeds_2 = self._get_clip_prompt_embeds(
                self.tokenizer_2, 
                self.text_encoder_2, 
                negative_prompt_2,
                num_images_per_prompt = num_images_per_prompt,
                max_sequence_length = max_sequence_length,
                device = device,
                dtype = dtype,
            )

            if negative_pooled_prompt_embeds_2.shape[0] == 1 and batch_size > 1:
                negative_pooled_prompt_embeds_2 = negative_pooled_prompt_embeds_2.repeat(batch_size, 1)

        if pooled_prompt_embeds is None:
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)

        if do_classifier_free_guidance and negative_pooled_prompt_embeds is None:
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds_1, negative_pooled_prompt_embeds_2], dim=-1
            )

        if prompt_embeds_t5 is None:
            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            if len(prompt_3) > 1 and len(prompt_3) != batch_size:
                raise ValueError(f"prompt_3 must be of length 1 or {batch_size}")

            prompt_embeds_t5 = self._get_t5_prompt_embeds(
                prompt_3,
                num_images_per_prompt = num_images_per_prompt,
                max_sequence_length = max_sequence_length,
                device = device,
                dtype = dtype,
                )

            if prompt_embeds_t5.shape[0] == 1 and batch_size > 1:
                prompt_embeds_t5 = prompt_embeds_t5.repeat(batch_size, 1, 1)

        if do_classifier_free_guidance and negative_prompt_embeds_t5 is None:
            negative_prompt_3 = negative_prompt_3 or negative_prompt
            negative_prompt_3 = [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3

            if len(negative_prompt_3) > 1 and len(negative_prompt_3) != batch_size:
                raise ValueError(f"negative_prompt_3 must be of length 1 or {batch_size}")

            negative_prompt_embeds_t5 = self._get_t5_prompt_embeds(
                negative_prompt_3, 
                num_images_per_prompt = num_images_per_prompt,
                max_sequence_length = max_sequence_length,
                device = device,
                dtype = dtype,
            )

            if negative_prompt_embeds_t5.shape[0] == 1 and batch_size > 1:
                negative_prompt_embeds_t5 = negative_prompt_embeds_t5.repeat(batch_size, 1, 1)

        if prompt_embeds_llama3 is None:
            prompt_4 = prompt_4 or prompt
            prompt_4 = [prompt_4] if isinstance(prompt_4, str) else prompt_4

            if len(prompt_4) > 1 and len(prompt_4) != batch_size:
                raise ValueError(f"prompt_4 must be of length 1 or {batch_size}")

            prompt_embeds_llama3 = self._get_llama3_prompt_embeds(
                prompt_4, 
                num_images_per_prompt = num_images_per_prompt,
                max_sequence_length = max_sequence_length,
                device = device,
                dtype = dtype,
                )

            if prompt_embeds_llama3.shape[0] == 1 and batch_size > 1:
                prompt_embeds_llama3 = prompt_embeds_llama3.repeat(1, batch_size, 1, 1)

        if do_classifier_free_guidance and negative_prompt_embeds_llama3 is None:
            negative_prompt_4 = negative_prompt_4 or negative_prompt
            negative_prompt_4 = [negative_prompt_4] if isinstance(negative_prompt_4, str) else negative_prompt_4

            if len(negative_prompt_4) > 1 and len(negative_prompt_4) != batch_size:
                raise ValueError(f"negative_prompt_4 must be of length 1 or {batch_size}")

            negative_prompt_embeds_llama3 = self._get_llama3_prompt_embeds(
                negative_prompt_4, 
                num_images_per_prompt = num_images_per_prompt,
                max_sequence_length = max_sequence_length,
                device = device,
                dtype = dtype,
            )

            if negative_prompt_embeds_llama3.shape[0] == 1 and batch_size > 1:
                negative_prompt_embeds_llama3 = negative_prompt_embeds_llama3.repeat(1, batch_size, 1, 1)

        # duplicate pooled_prompt_embeds for each generation per prompt
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        # duplicate t5_prompt_embeds for batch_size and num_images_per_prompt
        bs_embed, seq_len, _ = prompt_embeds_t5.shape
        if bs_embed == 1 and batch_size > 1:
            prompt_embeds_t5 = prompt_embeds_t5.repeat(batch_size, 1, 1)
        elif bs_embed > 1 and bs_embed != batch_size:
            raise ValueError(f"cannot duplicate prompt_embeds_t5 of batch size {bs_embed}")
        prompt_embeds_t5 = prompt_embeds_t5.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_t5 = prompt_embeds_t5.view(batch_size * num_images_per_prompt, seq_len, -1)

        # duplicate llama3_prompt_embeds for batch_size and num_images_per_prompt
        _, bs_embed, seq_len, dim = prompt_embeds_llama3.shape
        if bs_embed == 1 and batch_size > 1:
            prompt_embeds_llama3 = prompt_embeds_llama3.repeat(1, batch_size, 1, 1)
        elif bs_embed > 1 and bs_embed != batch_size:
            raise ValueError(f"cannot duplicate prompt_embeds_llama3 of batch size {bs_embed}")
        prompt_embeds_llama3 = prompt_embeds_llama3.repeat(1, 1, num_images_per_prompt, 1)
        prompt_embeds_llama3 = prompt_embeds_llama3.view(-1, batch_size * num_images_per_prompt, seq_len, dim)

        if do_classifier_free_guidance:
            # duplicate negative_pooled_prompt_embeds for batch_size and num_images_per_prompt
            bs_embed, seq_len = negative_pooled_prompt_embeds.shape
            if bs_embed == 1 and batch_size > 1:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(batch_size, 1)
            elif bs_embed > 1 and bs_embed != batch_size:
                raise ValueError(f"cannot duplicate negative_pooled_prompt_embeds of batch size {bs_embed}")
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

            # duplicate negative_t5_prompt_embeds for batch_size and num_images_per_prompt
            bs_embed, seq_len, _ = negative_prompt_embeds_t5.shape
            if bs_embed == 1 and batch_size > 1:
                negative_prompt_embeds_t5 = negative_prompt_embeds_t5.repeat(batch_size, 1, 1)
            elif bs_embed > 1 and bs_embed != batch_size:
                raise ValueError(f"cannot duplicate negative_prompt_embeds_t5 of batch size {bs_embed}")
            negative_prompt_embeds_t5 = negative_prompt_embeds_t5.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds_t5 = negative_prompt_embeds_t5.view(batch_size * num_images_per_prompt, seq_len, -1)

            # duplicate negative_prompt_embeds_llama3 for batch_size and num_images_per_prompt
            _, bs_embed, seq_len, dim = negative_prompt_embeds_llama3.shape
            if bs_embed == 1 and batch_size > 1:
                negative_prompt_embeds_llama3 = negative_prompt_embeds_llama3.repeat(1, batch_size, 1, 1)
            elif bs_embed > 1 and bs_embed != batch_size:
                raise ValueError(f"cannot duplicate negative_prompt_embeds_llama3 of batch size {bs_embed}")
            negative_prompt_embeds_llama3 = negative_prompt_embeds_llama3.repeat(1, 1, num_images_per_prompt, 1)
            negative_prompt_embeds_llama3 = negative_prompt_embeds_llama3.view(
                -1, batch_size * num_images_per_prompt, seq_len, dim
            )

        return (
            prompt_embeds_t5,
            negative_prompt_embeds_t5,
            prompt_embeds_llama3,
            negative_prompt_embeds_llama3,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_inpaint.StableDiffusion3InpaintPipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i: i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def _pack_latent(self, latents, batch_size):
        pH, pW = latents.shape[-2] // self.transformer.config.patch_size, latents.shape[-1] // self.transformer.config.patch_size
        x = einops.rearrange(x, 'B C (H p1) (W p2) -> B (H W) (p1 p2 C)', p1=self.config.patch_size, p2=self.config.patch_size)
        img_sizes = [[pH, pW]] * batch_size
        return x, img_sizes

    # TODO - Added condition_image and also subject_image
    # Total Added attributes: 1. subject_image 2. condition_image, latents=none, cond_number=1, sub_number=1
    # Refer this - https://github.com/Xiaojiu-z/EasyControl/blob/351ff8278e7a7f1fe4ba7bd98814d1bea401ef06/src/pipeline.py#L425
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        subject_image,
        condition_image,
        latents=None,
        cond_number=1,
        sub_number=1
    ):
        

        """
        HiDreamImage latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        by the patch size. So the vae scale factor is multiplied by the patch size to account for this.
        """
        # Scale the latent height of conditioning image
        height_cond = 2 * (self.cond_size // (self.vae_scale_factor * 2))
        width_cond = 2 * (self.cond_size // (self.vae_scale_factor * 2))

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)
        
        # latent
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        

        latents_to_concat, latents_ids_to_concat = [], []
        # subject
        if subject_image is not None:
            shape_subject = (batch_size, num_channels_latents, height_cond*sub_number, width_cond)
            subject_image = subject_image.to(device=device, dtype=dtype)
            subject_image_latents = self._encode_vae_image(image=subject_image, generator=generator)
            latents_to_concat.append(subject_image_latents.unsqueeze(0))
            latent_image_ids = torch.zeros(height_cond // 2, width_cond // 2, 3, device=device, dtype=dtype)
            latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height_cond // 2, device=device)[:, None] 
            latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width_cond // 2, device=device)[None, :]
            latent_image_ids[:,1] += + 64 # fixed offset
            latent_image_ids = repeat(latent_image_ids, "h w c -> b (h w) c", b=batch_size)
            subject_latent_image_ids = torch.concat([latent_image_ids for _ in range(sub_number)], dim=-2)
            latents_ids_to_concat.append(subject_latent_image_ids)

        # spatial
        if condition_image is not None:
            shape_cond = (batch_size, num_channels_latents, height_cond*cond_number, width_cond)  
            condition_image = condition_image.to(device=device, dtype=dtype)
            cond_image_latents = self._encode_vae_image(image=condition_image, generator=generator)
            latents_to_concat.append(cond_image_latents.unsqueeze(0))
            scale_h = height / height_cond
            scale_w = width / width_cond
            latent_image_ids = torch.zeros(height_cond // 2, width_cond // 2, 3, device=device, dtype=dtype)
            latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height_cond // 2, device=device)[:, None] * scale_h
            latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width_cond // 2, device=device)[None, :] * scale_w
            latent_image_ids = repeat(latent_image_ids, "h w c -> b (h w) c", b=batch_size)
            cond_latent_image_ids = torch.concat([latent_image_ids for _ in range(cond_number)], dim=-2)
            latents_ids_to_concat.append(cond_latent_image_ids)
            

        # Concatenate or return Non
        cond_latents = torch.cat(latents_to_concat, dim=0) if latents_to_concat else None
        cond_img_id = torch.cat(latents_ids_to_concat, dim=-2) if latents_ids_to_concat else None
        return latents, cond_latents, cond_img_id
    
    @property
    def guidance_scale(self):
        return self._guidance_scale
    
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1
    
    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs
    
    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt
    
    # TODO - Add variable for conditioning 1. spatial_image=[], 2. subject_image=[], 3. cond_size=512
    # Refer - https://github.com/Xiaojiu-z/EasyControl/blob/351ff8278e7a7f1fe4ba7bd98814d1bea401ef06/src/pipeline.py#L509
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_4: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        negative_prompt_4: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 128,
        spatial_images=[],
        subject_images=[],
        cond_size=512,
    ):
        
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self.cond_size = cond_size

        division = self.vae_scale_factor * 2
        S_max = (self.default_sample_size * self.vae_scale_factor) ** 2
        scale = S_max / (width * height)
        scale = math.sqrt(scale)
        width, height = int(width * scale // division * division), int(height * scale // division * division)

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        cond_number = len(spatial_images)
        sub_number = len(subject_images)

        if sub_number > 0:
            subject_image_ls = []
            for subject_image in subject_images:
                w, h = subject_image.size[:2]
                scale = self.cond_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                subject_image = self.image_processor.preprocess(subject_image, height=new_h, width=new_w)
                subject_image = subject_image.to(dtype=torch.float32)
                pad_h = cond_size - subject_image.shape[-2]
                pad_w = cond_size - subject_image.shape[-1]
                subject_image = pad(
                    subject_image,
                    padding=(int(pad_w / 2), int(pad_h / 2), int(pad_w / 2), int(pad_h / 2)),
                    fill=0
                )
                subject_image_ls.append(subject_image)
            subject_image = torch.concat(subject_image_ls, dim=-2)
        else:
            subject_image = None

        if cond_number > 0:
            condition_image_ls = []
            for img in spatial_images:
                print(img)
                condition_image = self.image_processor.preprocess(img, height=self.cond_size, width=self.cond_size)
                condition_image = condition_image.to(dtype=torch.float32)
                condition_image_ls.append(condition_image)
            condition_image = torch.concat(condition_image_ls, dim=-2)
        else:
            condition_image = None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            prompt_4=prompt_4,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            negative_prompt_4=negative_prompt_4,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds_arr = []
            for n, p in zip(negative_prompt_embeds, prompt_embeds):
                if len(n.shape) == 3:
                    prompt_embeds_arr.append(torch.cat([n, p], dim=0))
                else:
                    prompt_embeds_arr.append(torch.cat([n, p], dim=1))
            prompt_embeds = prompt_embeds_arr
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        # TODO figure out the outputs of the function
        latents, cond_latents, cond_img_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            pooled_prompt_embeds.dtype,
            device,
            generator,
            subject_image,
            condition_image,
            latents,
            cond_number,
            sub_number
        )

        """
        When (H == W) the img_ids are not calculated. More on the topic: https://chatgpt.com/s/dr_681dcec9e3b08191bb43194e45f02117
        For this reason we are not calculating the cond_ids since the condition images will always be square.
        """

        if latents.shape[-2] != latents.shape[-1]:
            B, C, H, W = latents.shape
            pH, pW = H // self.transformer.config.patch_size, W // self.transformer.config.patch_size

            img_sizes = torch.tensor([pH, pW], dtype=torch.int64).reshape(-1)
            img_ids = torch.zeros(pH, pW, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW)[None, :]
            img_ids = img_ids.reshape(pH * pW, -1)
            img_ids_pad = torch.zeros(self.transformer.max_seq, 3)
            img_ids_pad[:pH*pW, :] = img_ids

            img_sizes = img_sizes.unsqueeze(0).to(latents.device) 
            img_ids = img_ids_pad.unsqueeze(0).to(latents.device) 
            if self.do_classifier_free_guidance:
                img_sizes = img_sizes.repeat(2 * B, 1)
                img_ids = img_ids.repeat(2 * B, 1, 1)
        else:
            img_sizes = img_ids = None

        # 5. Prepare timesteps
        mu = calculate_shift(self.transformer.max_seq)
        scheduler_kwargs = {"mu": mu}
        if isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=math.exp(mu))
            timesteps = self.scheduler.timesteps
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                **scheduler_kwargs,
            )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                if latent_model_input.shape[-2] != latent_model_input.shape[-1]:
                    B, C, H, W = latent_model_input.shape
                    patch_size = self.transformer.config.patch_size
                    pH, pW = H // patch_size, W // patch_size
                    out = torch.zeros(
                        (B, C, self.transformer.max_seq, patch_size * patch_size), 
                        dtype=latent_model_input.dtype, 
                        device=latent_model_input.device
                    )
                    latent_model_input = einops.rearrange(latent_model_input, 'B C (H p1) (W p2) -> B C (H W) (p1 p2)', p1=patch_size, p2=patch_size)
                    out[:, :, 0:pH*pW] = latent_model_input
                    latent_model_input = out

                noise_pred = self.transformer(
                    hidden_states = latent_model_input,
                    cond_hidden_states=cond_latents,
                    cond_img_ids=cond_img_ids,
                    timesteps = timestep,
                    encoder_hidden_states = prompt_embeds,
                    pooled_embeds = pooled_prompt_embeds,
                    img_sizes = img_sizes,
                    img_ids = img_ids,
                    return_dict = False,
                )[0]
                noise_pred = -noise_pred

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return HiDreamImagePipelineOutput(images=image)