import inspect
import math
from typing import Callable, List, Optional, Tuple, Union
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from diffusers.models.attention_processor import Attention
from .attention import HiDreamAttention
from .attention_processor import attention, apply_rope

class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        cond_width=512,
        cond_height=512,
        number=0,
        n_loras=1
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)
        
        self.cond_height = cond_height
        self.cond_width = cond_width
        self.number = number
        self.n_loras = n_loras

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        #### img condition
        batch_size = hidden_states.shape[0]
        cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64
        block_size =  hidden_states.shape[1] - cond_size * self.n_loras
        shape = (batch_size, hidden_states.shape[1], hidden_states.shape[2])
        mask = torch.ones(shape, device=hidden_states.device, dtype=dtype) 
        mask[:, :block_size+self.number*cond_size, :] = 0
        mask[:, block_size+(self.number+1)*cond_size:, :] = 0
        hidden_states = mask * hidden_states
        ####
        
        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)
    

class MultiSingleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, ranks=[], lora_weights=[], network_alphas=[], device=None, dtype=None, cond_width=512, cond_height=512, n_loras=1):
        super().__init__()
        # Initialize a list to store the LoRA layers
        self.n_loras = n_loras
        self.cond_width = cond_width
        self.cond_height = cond_height
        
        self.q_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i],network_alphas[i], device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i],network_alphas[i], device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i],network_alphas[i], device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.proj_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i],network_alphas[i], device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.lora_weights = lora_weights
        
    def __call__(
        self,
        attn: HiDreamAttention,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
        use_cond = False,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        
        # Initialize basic parameters
        dtype = image_tokens.dtype
        batch_size = image_tokens.shape[0]

        # Process tokens through main attention projections
        query_i = attn.to_q(image_tokens)
        key_i = attn.to_k(image_tokens)
        value_i = attn.to_v(image_tokens)

        # Apply LoRA layers and combine with base projections
        for i in range(self.n_loras):
            query = query_i + self.lora_weights[i] * self.q_loras[i](image_tokens)
            key = key_i + self.lora_weights[i] * self.k_loras[i](image_tokens)
            value = value_i + self.lora_weights[i] * self.v_loras[i](image_tokens)

        # Apply normalization
        query_i = attn.q_rms_norm(query_i).to(dtype=dtype)
        key_i = attn.k_rms_norm(key_i).to(dtype=dtype)

        # Reshape for multi-head attention
        inner_dim = key_i.shape[-1]
        head_dim = inner_dim // attn.heads
        query_i = query_i.view(batch_size, -1, attn.heads, head_dim)
        key_i = key_i.view(batch_size, -1, attn.heads, head_dim)
        value_i = value_i.view(batch_size, -1, attn.heads, head_dim)
        if image_tokens_masks is not None:  # Apply masks if provided
            key_i = key_i * image_tokens_masks.view(batch_size, -1, 1, 1)

        query = query_i
        key = key_i
        value = value_i

        # Apply rotary positional embeddings (RoPE)
        if query.shape[-1] == rope.shape[-3] * 2:
            query, key = apply_rope(query, key, rope)
        else:
            query_1, query_2 = query.chunk(2, dim=-1)
            key_1, key_2 = key.chunk(2, dim=-1)
            query_1, key_1 = apply_rope(query_1, key_1, rope)
            query = torch.cat([query_1, query_2], dim=-1)
            key = torch.cat([key_1, key_2], dim=-1)
        
        # Calculate sizes for attention masking
        cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64
        block_size =  image_tokens.shape[1] - cond_size * self.n_loras
        scaled_cond_size = cond_size
        scaled_block_size = block_size
        scaled_seq_len = query.shape[2]
        
        # Create attention mask for conditional blocks
        num_cond_blocks = self.n_loras
        mask = torch.ones((scaled_seq_len, scaled_seq_len), device=image_tokens.device)
        mask[ :scaled_block_size, :] = 0  # First block_size row
        for i in range(num_cond_blocks):
            start = i * scaled_cond_size + scaled_block_size
            end = (i + 1) * scaled_cond_size + scaled_block_size
            mask[start:end, start:end] = 0  # Diagonal blocks
        mask = mask * -1e20
        mask = mask.to(query.dtype)


        # Compute attention with mask
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=mask)
        hidden_states = hidden_states.flatten(-2)
        hidden_states = hidden_states.to(query.dtype)
        # Apply output projection and LoRA
        hidden_states = attn.to_out(hidden_states)
        for i in range(self.n_loras):
                    hidden_states = hidden_states + self.lora_weights[i] * self.proj_loras[i](hidden_states) 
        
        # Split output into main and conditional hidden states
        cond_hidden_states = hidden_states[:, block_size:,:]
        hidden_states = hidden_states[:, : block_size,:]
        
        if use_cond:
            return hidden_states, cond_hidden_states
        else:
            return hidden_states
 
class MultiDoubleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, ranks=[], lora_weights=[], network_alphas=[], device=None, dtype=None, cond_width=512, cond_height=512, n_loras=1):
        super().__init__()
        
        # Initialize a list to store the LoRA layers
        self.n_loras = n_loras
        self.cond_width = cond_width
        self.cond_height = cond_height
        self.q_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i],network_alphas[i], device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i],network_alphas[i], device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i],network_alphas[i], device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.proj_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i],network_alphas[i], device=device, dtype=dtype, cond_width=cond_width, cond_height=cond_height, number=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.lora_weights = lora_weights

    def __call__(
        self,
        attn: HiDreamAttention,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
        use_cond = False,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        
        # Initialize basic parameters
        dtype = image_tokens.dtype
        batch_size = image_tokens.shape[0]

        # Process image tokens through main attention projections
        query_i = attn.to_q(image_tokens)
        key_i = attn.to_k(image_tokens)
        value_i = attn.to_v(image_tokens)

        # Apply LoRA layers to image token projections
        for i in range(self.n_loras):
            query_i = query_i + self.lora_weights[i] * self.q_loras[i](image_tokens)
            key_i = key_i + self.lora_weights[i] * self.k_loras[i](image_tokens)
            value_i = value_i + self.lora_weights[i] * self.v_loras[i](image_tokens)

        # Apply normalization
        query_i = attn.q_rms_norm(query_i).to(dtype=dtype)
        key_i = attn.k_rms_norm(key_i).to(dtype=dtype)

        # Calculate dimensions for multi-head attention
        inner_dim = key_i.shape[-1]
        head_dim = inner_dim // attn.heads
        # Reshape for multi-head attention
        query_i = query_i.view(batch_size, -1, attn.heads, head_dim)
        key_i = key_i.view(batch_size, -1, attn.heads, head_dim)
        value_i = value_i.view(batch_size, -1, attn.heads, head_dim)
        if image_tokens_masks is not None:  # Apply masks if provided
            key_i = key_i * image_tokens_masks.view(batch_size, -1, 1, 1)

        # Process text tokens through text-specific attention projections
        query_t = attn.q_rms_norm_t(attn.to_q_t(text_tokens)).to(dtype=dtype)
        key_t = attn.k_rms_norm_t(attn.to_k_t(text_tokens)).to(dtype=dtype)
        value_t = attn.to_v_t(text_tokens)

        # Reshape text token tensors for multi-head attention
        query_t = query_t.view(batch_size, -1, attn.heads, head_dim)
        key_t = key_t.view(batch_size, -1, attn.heads, head_dim)
        value_t = value_t.view(batch_size, -1, attn.heads, head_dim)

        # Concatenate image and text tensors
        num_image_tokens = query_i.shape[1]
        num_text_tokens = query_t.shape[1]
        query = torch.cat([query_i, query_t], dim=1)
        key = torch.cat([key_i, key_t], dim=1)
        value = torch.cat([value_i, value_t], dim=1)

        # Apply rotary positional embeddings (RoPE)
        if query.shape[-1] == rope.shape[-3] * 2:
            query, key = apply_rope(query, key, rope)
        else:
            query_1, query_2 = query.chunk(2, dim=-1)
            key_1, key_2 = key.chunk(2, dim=-1)
            query_1, key_1 = apply_rope(query_1, key_1, rope)
            query = torch.cat([query_1, query_2], dim=-1)
            key = torch.cat([key_1, key_2], dim=-1)

        # Calculate sizes for attention mask
        cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64 
        block_size =  image_tokens.shape[1] - cond_size * self.n_loras
        scaled_cond_size = cond_size
        scaled_seq_len = query.shape[2] #TODO - this doesn't look right 
        scaled_block_size = scaled_seq_len - cond_size * self.n_loras
        
        # Create attention mask for conditional blocks
        num_cond_blocks = self.n_loras
        mask = torch.ones((scaled_seq_len, scaled_seq_len), device=image_tokens.device)
        mask[ :scaled_block_size, :] = 0  # First block_size row
        for i in range(num_cond_blocks):
            start = i * scaled_cond_size + scaled_block_size
            end = (i + 1) * scaled_cond_size + scaled_block_size
            mask[start:end, start:end] = 0  # Diagonal blocks
        mask = mask * -1e20
        mask = mask.to(query.dtype)

        # Compute attention with mask
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=mask)
        hidden_states = hidden_states.flatten(-2)
        hidden_states = hidden_states.to(query.dtype)
        # Split hidden states back into image and text components
        hidden_states_i, hidden_states_t = torch.split(hidden_states, [num_image_tokens, num_text_tokens], dim=1)


        # Apply output projections and LoRA layers
        hidden_states_i = attn.to_out(hidden_states_i)
        for i in range(self.n_loras):
                    hidden_states_i = hidden_states_i + self.lora_weights[i] * self.proj_loras[i](hidden_states_i) 
        
        # Final processing of text and conditional outputs
        hidden_states_t = attn.to_out_t(hidden_states_t)
        cond_hidden_states = hidden_states_i[:, block_size:,:]
        hidden_states_i = hidden_states_i[:, :block_size,:]
        
        return hidden_states_i, hidden_states_t, cond_hidden_states