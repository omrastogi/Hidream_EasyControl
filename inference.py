import torch
import argparse
from PIL import Image
from hi_diffusers import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="dev")
args = parser.parse_args()
model_type = args.model_type
MODEL_PREFIX = "HiDream-ai"
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
from hi_diffusers.models.lora_helper import MultiDoubleStreamBlockLoraProcessor, MultiSingleStreamBlockLoraProcessor
 

# Model configurations
MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    }
}

# Resolution options
RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]

# Load models
def load_models(model_type):
    config = MODEL_CONFIGS[model_type]
    pretrained_model_name_or_path = config["path"]
    scheduler = MODEL_CONFIGS[model_type]["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)
    
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
        LLAMA_MODEL_NAME,
        use_fast=False)
    
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16).to("cuda")

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="transformer", 
        torch_dtype=torch.bfloat16).to("cuda")

    pipe = HiDreamImagePipeline.from_pretrained(
        pretrained_model_name_or_path, 
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16
    ).to("cuda", torch.bfloat16)
    pipe.transformer = transformer
    
    return pipe, config

# Parse resolution string to get height and width
def parse_resolution(resolution_str):
    if "1024 × 1024" in resolution_str:
        return 1024, 1024
    elif "768 × 1360" in resolution_str:
        return 768, 1360
    elif "1360 × 768" in resolution_str:
        return 1360, 768
    elif "880 × 1168" in resolution_str:
        return 880, 1168
    elif "1168 × 880" in resolution_str:
        return 1168, 880
    elif "1248 × 832" in resolution_str:
        return 1248, 832
    elif "832 × 1248" in resolution_str:
        return 832, 1248
    else:
        return 1024, 1024  # Default fallback

# Generate image function
def generate_image(pipe, model_type, prompt, resolution, seed):
    # Get configuration for current model
    config = MODEL_CONFIGS[model_type]
    guidance_scale = config["guidance_scale"]
    num_inference_steps = config["num_inference_steps"]
    
    # Parse resolution
    height, width = parse_resolution(resolution)
    
    # Handle seed
    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()
    
    generator = torch.Generator("cuda").manual_seed(seed)

    use_cond = True

    if use_cond:
        prompt = "A SKS on the car"
        # subject_images = [Image.open("./test_imgs/subject_1.png").convert("RGB")]
        spatial_images = [Image.open("./test_imgs/inpainting.png").convert("RGB")]
    else:
        prompt = "A SKS on the car"
        subject_images = []
        spatial_images = []
    # subject_images = []
    images = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
        generator=generator,
        subject_images=[],
        spatial_images=spatial_images,
        cond_size=512,        
    ).images
    
    return images[0], seed

# Initialize with default model
print("Loading default model (full)...")

pipe, _ = load_models(model_type)
use_cond = True
##############################
if use_cond:
    lora_attn_procs = {}
    import re
    number = 1
    ranks = [64]
    cond_size = 512
    double_blocks_idx = list(range(16))
    single_blocks_idx = list(range(32))
    for name, attn_processor in pipe.transformer.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))
        if name.startswith("double_stream_blocks") and layer_index in double_blocks_idx:
            print("setting LoRA Processor for", name)
            lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                        dim=2560, ranks=ranks, network_alphas=ranks, lora_weights=[1 for _ in range(number)], device=pipe.device, dtype=pipe.dtype, cond_width=cond_size, cond_height=cond_size, n_loras=number
                    )
        elif name.startswith("single_stream_blocks") and layer_index in single_blocks_idx:
            print("setting LoRA Processor for", name)
            lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                        dim=2560, ranks=ranks, network_alphas=ranks, lora_weights=[1 for _ in range(number)], device=pipe.device, dtype=pipe.dtype, cond_width=cond_size, cond_height=cond_size, n_loras=number
                    )
        else:
            lora_attn_procs[name] = attn_processor
    pipe.transformer.set_attn_processor(lora_attn_procs)
##################################

from safetensors.torch import load_file
checkpoint_path = "/mnt/data/om/experiment-001/checkpoint-step00001100/model.safetensors"

state_dict = load_file(checkpoint_path)
pipe.transformer.load_state_dict(state_dict, strict=False)
# No need to reload or reinitialize state_dict; just use the one already loaded above.
del state_dict

print("Model loaded successfully!")
prompt = "A cat holding a sign that says \"Hi-Dreams.ai\"." 
resolution = "1024 × 1024 (Square)"
seed = -1
image, seed = generate_image(pipe, model_type, prompt, resolution, seed)
image.save("output.png")
