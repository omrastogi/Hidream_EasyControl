import argparse
import copy
import json
import logging
import math
import os
import shutil
from contextlib import nullcontext
from einops import repeat
from pathlib import Path
import re
from safetensors.torch import save_file

from PIL import Image
import numpy as np
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, FullyShardedDataParallelPlugin, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from hi_diffusers.models.attention_processor import HiDreamAttnProcessor_flashattn
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTokenizer, LlamaForCausalLM, PretrainedConfig, T5Tokenizer
import gc
import diffusers
from diffusers.training_utils import free_memory
from diffusers import (
    BitsAndBytesConfig,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
    convert_unet_state_dict_to_peft
)

from hi_diffusers.models.lora_helper import load_checkpoint
from hi_diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
from hi_diffusers.models.layers import MultiDoubleStreamBlockLoraProcessor, MultiSingleStreamBlockLoraProcessor
from hi_diffusers.models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel
from dataset.inpainting_dataset import StreamingImageCaptionInpaintDataset, custom_collate_fn

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)

def log_validation(
        pipeline,
        args,
        accelerator,
        pipeline_args,
        step,
        torch_dtype,
        is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    # autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
    autocast_ctx = nullcontext()

    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, step, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection" or model_class == "CLIPTextModel":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--lora_num", type=int, default=1, help="number of the lora.")
    parser.add_argument("--cond_size", type=int, default=512, help="size of the condition data.")
    parser.add_argument("--noise_size", type=int, default=1280, help="max side of the training data.")
    parser.add_argument("--test_h", type=int, default=1024, help="max side of the training data.")
    parser.add_argument("--test_w", type=int, default=1024, help="max side of the training data.")
    parser.add_argument("--mode",type=str,default=None,help="The mode of the controller. Choose between ['depth', 'pose', 'canny'].")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="",
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_tokenizer_4_name_or_path",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_text_encoder_4_name_or_path",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--bnb_quantization_config_path",
        type=str,
        default=None,
        help="Quantization config in a JSON file that will be used to define the bitsandbytes quant config of the DiT.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--spatial_column",
        type=str,
        default="None",
        help="The column of the dataset containing the canny image. By "
             "default, the standard Image Dataset maps out 'file_name' "
             "to 'image'.",
    )
    parser.add_argument(
        "--subject_column",
        type=str,
        default="image",
        help="The column of the dataset containing the subject image. By "
             "default, the standard Image Dataset maps out 'file_name' "
             "to 'image'.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
             "default, the standard Image Dataset maps out 'file_name' "
             "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption_left,caption_right",
        help="The column of the dataset containing the instance prompt for each image",
    )
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="A woodenpot floating in a pool.",
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--subject_test_images",
        type=str,
        nargs="+",
        default=["/tiamat-NAS/zhangyuxuan/datasets/benchmark_dataset/decoritems_woodenpot/0.png"],
        help="A list of subject test image paths.",
    )
    parser.add_argument(
        "--spatial_test_images",
        type=str,
        nargs="+",
        default=[],
        help="A list of spatial test image paths.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=20,
        help=(
            "Run validation every X epochs. validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        default=False,
        help="Whether to offload the VAE and the text encoder to CPU when they are not used.",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[4],
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alphas",
        type=int,
        nargs="+",
        default=[4],
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tiamat-NAS/zhangyuxuan/projects2/Easy_Control_0120/single_models/subject_model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
             "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
             "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

def compute_text_embeddings(prompt, text_encoding_pipeline):
    with torch.no_grad():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = text_encoding_pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            prompt_4=prompt,
            max_sequence_length=args.max_sequence_length)
    return (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    )

def load_text_encoders(class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant)
    text_encoder_two = class_two.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant)
    text_encoder_three = class_three.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant)
    text_encoder_four = LlamaForCausalLM.from_pretrained(
        args.pretrained_text_encoder_4_name_or_path,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16,
    )
    return text_encoder_one, text_encoder_two, text_encoder_three, text_encoder_four

def prepare_latent_image_ids_(height, width, device, dtype):
    latent_image_ids = torch.zeros(height//2, width//2, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height//2, device=device)[:, None]  # y
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width//2, device=device)[None, :]   # x
    return latent_image_ids

def resize_position_encoding(batch_size, original_height, original_width, target_height, target_width, device, dtype):
    latent_image_ids = prepare_latent_image_ids_(original_height, original_width, device, dtype)
    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )
    
    scale_h = original_height / target_height
    scale_w = original_width / target_width
    latent_image_ids_resized = torch.zeros(target_height//2, target_width//2, 3, device=device, dtype=dtype)
    latent_image_ids_resized[..., 1] = latent_image_ids_resized[..., 1] + torch.arange(target_height//2, device=device)[:, None] * scale_h
    latent_image_ids_resized[..., 2] = latent_image_ids_resized[..., 2] + torch.arange(target_width//2, device=device)[None, :] * scale_w
    
    cond_latent_image_id_height, cond_latent_image_id_width, cond_latent_image_id_channels = latent_image_ids_resized.shape
    cond_latent_image_ids = latent_image_ids_resized.reshape(
            cond_latent_image_id_height * cond_latent_image_id_width, cond_latent_image_id_channels
        )
    return latent_image_ids, cond_latent_image_ids 

def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.logging_dir, exist_ok=True)
    logging_dir = Path(args.output_dir, args.logging_dir)

    if args.subject_column == "None":
        args.subject_column = None
    if args.spatial_column == "None":
        args.spatial_column = None

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # TODO - remove the tokenizers
    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    tokenizer_four = AutoTokenizer.from_pretrained(
        args.pretrained_tokenizer_4_name_or_path,
        revision=args.revision,
    )
    tokenizer_four.pad_token = tokenizer_four.eos_token

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2")
    text_encoder_cls_three = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3")


    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three, text_encoder_four = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    quantization_config = None
    if args.bnb_quantization_config_path is not None:
        with open(args.bnb_quantization_config_path) as f:
            config_kwargs = json.load(f)
            if "load_in_4bit" in config_kwargs and config_kwargs["load_in_4bit"]:
                config_kwargs["bnb_4bit_compute_dtype"] = weight_dtype
        quantization_config = BitsAndBytesConfig(**config_kwargs)

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="transformer", 
        revision=args.revision, 
        variant=args.variant,
        quantization_config=quantization_config,
    )

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    text_encoder_four.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    
    # vae and text_encoders can be offloaded if not being used
    to_kwargs = {"dtype": weight_dtype, "device": accelerator.device} if not args.offload else {"dtype": weight_dtype}
    vae.to(**to_kwargs)
    text_encoder_one.to(**to_kwargs)
    text_encoder_two.to(**to_kwargs)
    text_encoder_three.to(**to_kwargs)
    text_encoder_four.to(**to_kwargs)

    text_encoding_pipeline = HiDreamImagePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        text_encoder_3=text_encoder_three,
        tokenizer_3=tokenizer_three,
        text_encoder_4=text_encoder_four,
        tokenizer_4=tokenizer_four,
    )
    # we never offload the transformer to CPU, so we can just use the accelerator device
    transformer_to_kwargs = {"device": accelerator.device} if args.bnb_quantization_config_path is not None else {"device": accelerator.device, "dtype": weight_dtype}
    transformer.to(**transformer_to_kwargs)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    #### lora_layers ####
    if args.pretrained_lora_path is not None:
        lora_path = args.pretrained_lora_path
        checkpoint = load_checkpoint(lora_path)
        lora_attn_procs = {}
        double_blocks_idx = list(range(16))
        single_blocks_idx = list(range(32))
        number = 1
        for name, attn_processor in transformer.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            
            if name.startswith("double_stream_blocks") and layer_index in double_blocks_idx:
                lora_state_dicts = {}
                for key, value in checkpoint.items():
                    # Match based on the layer index in the key (assuming the key contains layer index)
                    if re.search(r'\.(\d+)\.', key):
                        checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                        if checkpoint_layer_index == layer_index and key.startswith("double_stream_blocks"):
                            lora_state_dicts[key] = value
                
                # print("setting LoRA Processor for", name)
                lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                    dim=2560, ranks=args.ranks, network_alphas=args.network_alphas, lora_weights=[1 for _ in range(args.lora_num)], device=accelerator.device, dtype=weight_dtype, cond_width=args.cond_size, cond_height=args.cond_size, n_loras=args.lora_num
                )
                
                # Load the weights from the checkpoint dictionary into the corresponding layers
                for n in range(number):
                    lora_attn_procs[name].q_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.down.weight', None)
                    lora_attn_procs[name].q_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.up.weight', None)
                    lora_attn_procs[name].k_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.down.weight', None)
                    lora_attn_procs[name].k_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.up.weight', None)
                    lora_attn_procs[name].v_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.down.weight', None)
                    lora_attn_procs[name].v_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.up.weight', None)
                    lora_attn_procs[name].proj_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.proj_loras.{n}.down.weight', None)
                    lora_attn_procs[name].proj_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.proj_loras.{n}.up.weight', None)
                
            elif name.startswith("single_stream_blocks") and layer_index in single_blocks_idx:
                
                lora_state_dicts = {}
                for key, value in checkpoint.items():
                    # Match based on the layer index in the key (assuming the key contains layer index)
                    if re.search(r'\.(\d+)\.', key):
                        checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                        if checkpoint_layer_index == layer_index and key.startswith("single_stream_blocks"):
                            lora_state_dicts[key] = value
                
                # print("setting LoRA Processor for", name)        
                lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                    dim=2560, ranks=args.ranks, network_alphas=args.network_alphas, lora_weights=[1 for _ in range(args.lora_num)], device=accelerator.device, dtype=weight_dtype, cond_width=args.cond_size, cond_height=args.cond_size, n_loras=args.lora_num
                )
                
                # Load the weights from the checkpoint dictionary into the corresponding layers
                for n in range(number):
                    lora_attn_procs[name].q_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.down.weight', None)
                    lora_attn_procs[name].q_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.up.weight', None)
                    lora_attn_procs[name].k_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.down.weight', None)
                    lora_attn_procs[name].k_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.up.weight', None)
                    lora_attn_procs[name].v_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.down.weight', None)
                    lora_attn_procs[name].v_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.up.weight', None)
                    lora_attn_procs[name].proj_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.proj_loras.{n}.down.weight', None)
                    lora_attn_procs[name].proj_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.proj_loras.{n}.up.weight', None)
            else:
                # TODO - change this 
                lora_attn_procs[name] = HiDreamAttnProcessor_flashattn()
    else:
        lora_attn_procs = {}
        double_blocks_idx = list(range(16))
        single_blocks_idx = list(range(32))
        for name, attn_processor in transformer.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            if name.startswith("double_stream_blocks") and layer_index in double_blocks_idx:
                print("setting LoRA Processor for", name)
                lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                    dim=2560, ranks=args.ranks, network_alphas=args.network_alphas, lora_weights=[1 for _ in range(args.lora_num)], device=accelerator.device, dtype=weight_dtype, cond_width=args.cond_size, cond_height=args.cond_size, n_loras=args.lora_num
                ).to(device=accelerator.device)
            elif name.startswith("single_stream_blocks") and layer_index in single_blocks_idx:
                print("setting LoRA Processor for", name)
                lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                    dim=2560, ranks=args.ranks, network_alphas=args.network_alphas, lora_weights=[1 for _ in range(args.lora_num)], device=accelerator.device, dtype=weight_dtype, cond_width=args.cond_size, cond_height=args.cond_size, n_loras=args.lora_num
                ).to(device=accelerator.device)
            else:
                lora_attn_procs[name] = attn_processor        
    ######################
    # Count if somehting is on CPU in lora_attn
    transformer.set_attn_processor(lora_attn_procs)
    
    transformer.train()

    for n, param in transformer.named_parameters():
        if '_lora' not in n:
            param.requires_grad = False

    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in transformer.parameters() if not p.requires_grad)
    print(f"Trainable params: {trainable/1e6:.3f}M | Frozen params: {frozen/1e6:.3f}M | Total: {(trainable+frozen)/1e6:.3f}M")


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        global_step = int(path.split("-")[-1])
        initial_global_step = global_step
    else:
        initial_global_step = 0
        global_step = 0
        first_epoch = 0

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    # Optimization parameters
    params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]
    transformer_parameters_with_lr = {"params": params_to_optimize, "lr": args.learning_rate}
    print(sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000, 'parameters')

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        [transformer_parameters_with_lr],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoader 
    from streaming import Stream, StreamingDataLoader
    from dataset.streaming_utils import make_streams
    from streaming.base.util import clean_stale_shared_memory
    import pandas as pd 
    import tempfile

    clean_stale_shared_memory()
    # Create streams
    # STREAMS = make_streams(
    #     local=["/mnt/dashtoon_data/data_hub/gallery-dl/pinterest_mds_shards/aspect-ratio-0.6"],
    #     # local=tempfile.mkdtemp(prefix="streaming_cache_"),
    #     choose=[10]
    # )
    mds_streams = []
    for stream in ["/mnt/dashtoon_data/data_hub/gallery-dl/pinterest_mds_shards/aspect-ratio-0.6"]:
        mds_streams.append(Stream(local=stream, proportion=None, repeat=None, choose=10))

    # Load caption metadata
    meta_path = "/mnt/dashtoon_data/data_hub/gallery-dl/pinterest_meta_caption_info_rewritten.jsonl"
    caption_meta = pd.read_json(meta_path, orient="records", lines=True).reset_index(drop=True)

    # Create dataset and dataloader
    train_dataset = StreamingImageCaptionInpaintDataset(
        streams=mds_streams,
        item_key="__key__",
        image_key="image", 
        caption_key="caption",
        caption_metadata=caption_meta,
        sdxl_conditioning=False,
        aspect_ratio="ASPECT_RATIO_1024",
        center_crop=False,
        irregular_mask_prob=1.0,
        box_mask_prob=1.0,
        switch_to_dumb_mask_prob=0.25,
        shuffle=True,
        batch_size=4,
        batching_method="per_stream"
    )

    train_dataloader = StreamingDataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=32,
        prefetch_factor=2,
        pin_memory=True,
        # collate_fn=custom_collate_fn
    )

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.resume_from_checkpoint:
        first_epoch = global_step // num_update_steps_per_epoch
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    # ! NOTE:
    # ! We do not pass `train_dataloader` to `accelerator.prepare` because it is a StreamingDataLoader and we get the following error:
    # ! RuntimeError: You can't use batches of different size with `dispatch_batches=True` or when using an `IterableDataset`.either pass `dispatch_batches=False` and
    # ! have each process fetch its own batch  or pass `split_batches=True`. By doing so, the main process will fetch a full batch and slice it into `num_processes`
    # ! batches for each process.
    transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    precompute_latents = args.cache_latents
    if precompute_latents:
        prompt_embeds_cache = []
        negative_prompt_embeds_cache = []
        pooled_prompt_embeds_cache = []
        negative_pooled_prompt_embeds_cache = []
        latents_cache = []
        # if args.offload:
        #     vae = vae.to(accelerator.device)
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                text_encoding_pipeline = text_encoding_pipeline.to(accelerator.device)
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = compute_text_embeddings(batch["prompt"], text_encoding_pipeline)
                # Handle prompt_embeds which can be a list of tensors or a single tensor
                if isinstance(prompt_embeds, list):
                    prompt_embeds = [p.to("cpu") for p in prompt_embeds]
                else:
                    prompt_embeds = prompt_embeds.to("cpu")

                # Handle negative_prompt_embeds which can be a list of tensors or a single tensor  
                if isinstance(negative_prompt_embeds, list):
                    negative_prompt_embeds = [n.to("cpu") for n in negative_prompt_embeds]
                else:
                    negative_prompt_embeds = negative_prompt_embeds.to("cpu")

                # Handle pooled_prompt_embeds which is a single tensor
                pooled_prompt_embeds = pooled_prompt_embeds.to("cpu")

                # Handle negative_pooled_prompt_embeds which is a single tensor
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to("cpu")

                prompt_embeds_cache.append(prompt_embeds)
                negative_prompt_embeds_cache.append(negative_prompt_embeds) 
                pooled_prompt_embeds_cache.append(pooled_prompt_embeds)
                negative_pooled_prompt_embeds_cache.append(negative_pooled_prompt_embeds)

    ## Not touching the VAE
    # move back to cpu before deleting to ensure memory is freed see: https://github.com/huggingface/diffusers/issues/11376#issue-3008144624
    # if args.offload or args.cache_latents:
    #     vae = vae.to("cpu")
    #     if args.cache_latents:
    #         del vae
    # move back to cpu before deleting to ensure memory is freed see: https://github.com/huggingface/diffusers/issues/11376#issue-3008144624
    text_encoding_pipeline = text_encoding_pipeline.to("cpu")
    del (
        text_encoder_one,
        text_encoder_two,
        text_encoder_three,
        text_encoder_four,
        tokenizer_two,
        tokenizer_three,
        tokenizer_four,
        text_encoding_pipeline,
    )
    free_memory()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "Easy_Control"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        print(timesteps)
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype) 
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    # some fixed parameters 
    vae_scale_factor = 16
    height_cond = 2 * (args.cond_size // vae_scale_factor)
    width_cond = 2 * (args.cond_size // vae_scale_factor)        
    offset = 64
    print(accelerator.gradient_accumulation_steps)
    # NOTE -  42% gpu being used, when code is here
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]

            with accelerator.accumulate(models_to_accumulate):    
                # NOTE - Text id not required 
                # TODO - understand this and add the cache unloading - DONE 
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
                    prompt_embeds_cache[step],
                    negative_prompt_embeds_cache[step], 
                    pooled_prompt_embeds_cache[step],
                    negative_pooled_prompt_embeds_cache[step]
                )
                # Handle prompt_embeds which can be a list of tensors or a single tensor
                if isinstance(prompt_embeds, list):
                    prompt_embeds = [p.to(accelerator.device) for p in prompt_embeds]
                else:
                    prompt_embeds = prompt_embeds.to(accelerator.device)

                # Handle negative_prompt_embeds which can be a list of tensors or a single tensor  
                if isinstance(negative_prompt_embeds, list):
                    negative_prompt_embeds = [n.to(accelerator.device) for n in negative_prompt_embeds]
                else:
                    negative_prompt_embeds = negative_prompt_embeds.to(accelerator.device)

                # Handle pooled_prompt_embeds which is a single tensor
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)

                # Handle negative_pooled_prompt_embeds which is a single tensor
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(accelerator.device)

                # Move vae to accelerator device for encoding
                vae = vae.to(accelerator.device)
                pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                
                height_ = 2 * (int(pixel_values.shape[-2]) // vae_scale_factor)
                width_ = 2 * (int(pixel_values.shape[-1]) // vae_scale_factor)

                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                # NOTE - 44% gpu 
                # ! NOTE - Hidream doesn't need any img_ids, as it has a patchify function 
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device, dtype=model_input.dtype)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # Handle non-square images
                if noisy_model_input.shape[-2] != noisy_model_input.shape[-1]:
                    B, C, H, W = noisy_model_input.shape
                    max_seq, patch_size = accelerator.unwrap_model(model=transformer).max_seq, accelerator.unwrap_model(model=transformer).config.patch_size
                    pH, pW = H // patch_size, W // patch_size

                    img_sizes = torch.tensor([pH, pW], dtype=torch.int64).reshape(-1)
                    img_ids = torch.zeros(pH, pW, 3)
                    img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH)[:, None]
                    img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW)[None, :]
                    img_ids = img_ids.reshape(pH * pW, -1)

                    img_ids_pad = torch.zeros(max_seq, 3)
                    img_ids_pad[: pH * pW, :] = img_ids

                    img_sizes = img_sizes.unsqueeze(0).to(noisy_model_input.device)
                    img_ids = img_ids_pad.unsqueeze(0).to(noisy_model_input.device)
                    img_sizes = img_sizes.repeat(B, 1)
                    img_ids = img_ids.repeat(B, 1, 1)

                    out = torch.zeros((B, C, max_seq, patch_size * patch_size), dtype=noisy_model_input.dtype, device=noisy_model_input.device)
                    from einops import rearrange
                    latent_model_input = rearrange(noisy_model_input, "b c (h p1) (w p2) -> b c (h w) (p1 p2)", p1=patch_size, p2=patch_size)
                    out[:, :, 0 : pH * pW] = latent_model_input
                    latent_model_input = out

                else:
                    img_sizes, img_ids = None, None
                    latent_model_input = noisy_model_input

                # ! NOTE: No packed latent required for HiDream
                cond_image_ids_to_concat = []
                latents_to_concat = []
                
                if args.subject_column is not None:
                    # Dealing with the latents
                    subject_pixel_values = batch["subject_pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                    subject_input = vae.encode(subject_pixel_values).latent_dist.sample()
                    subject_input = (subject_input - vae_config_shift_factor) * vae_config_scaling_factor
                    subject_input = subject_input.to(dtype=weight_dtype)
                    latents_to_concat.append(subject_input.unsqueeze(0))
                    # Dealing with the img_ids for subject
                    sub_number = subject_pixel_values.shape[-2] // args.cond_size
                    sub_latent_image_ids = torch.zeros(height_cond // 2, width_cond // 2, 3, device=model_input.device, dtype=weight_dtype)
                    sub_latent_image_ids[..., 1] = sub_latent_image_ids[..., 1] + torch.arange(height_cond // 2, device=model_input.device)[:, None]
                    sub_latent_image_ids[..., 2] = sub_latent_image_ids[..., 2] + torch.arange(width_cond // 2, device=model_input.device)[None, :]
                    sub_latent_image_ids = repeat(sub_latent_image_ids, "h w c -> b (h w) c", b=bsz)
                    sub_latent_image_ids[:, 1] += offset
                    sub_latent_image_ids = torch.concat([sub_latent_image_ids for _ in range(sub_number)], dim=-2)
                    cond_image_ids_to_concat.append(sub_latent_image_ids)
                    subject_pixel_values = subject_pixel_values.to("cpu")
                    del subject_pixel_values
                
                if args.spatial_column is not None:
                    # Dealing with the latents
                    cond_pixel_values = batch["cond_pixel_values"].to(device=accelerator.device, dtype=weight_dtype)             
                    cond_input = vae.encode(cond_pixel_values).latent_dist.sample()
                    cond_input = (cond_input - vae_config_shift_factor) * vae_config_scaling_factor
                    cond_input = cond_input.to(dtype=weight_dtype)
                    latents_to_concat.append(cond_input.unsqueeze(0))
                    # Dealing with the img_ids for cond
                    cond_number = cond_pixel_values.shape[-2] // args.cond_size
                    scale_h = height_ / height_cond
                    scale_w = width_ / width_cond
                    cond_latent_image_ids = torch.zeros(height_cond // 2, width_cond // 2, 3, device=model_input.device, dtype=weight_dtype)
                    cond_latent_image_ids[..., 1] = cond_latent_image_ids[..., 1] + torch.arange(height_cond // 2, device=model_input.device)[:, None] * scale_h
                    cond_latent_image_ids[..., 2] = cond_latent_image_ids[..., 2] + torch.arange(width_cond // 2, device=model_input.device)[None, :] * scale_w
                    cond_latent_image_ids = repeat(cond_latent_image_ids, "h w c -> b (h w) c", b=bsz)
                    cond_latent_image_ids = torch.concat([cond_latent_image_ids for _ in range(cond_number)], dim=-2)
                    cond_image_ids_to_concat.append(cond_latent_image_ids)
                    cond_pixel_values = cond_pixel_values.to("cpu")
                    del cond_pixel_values

                cond_image_ids = torch.concat(cond_image_ids_to_concat, dim=-2)
                cond_input = torch.cat(latents_to_concat, dim=0)
                pixel_values = pixel_values.to("cpu")
                vae.to("cpu")
                del pixel_values

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # Predict the noise residual
                    model_pred = transformer(
                        hidden_states=latent_model_input,
                        cond_hidden_states=cond_input,
                        cond_img_ids=cond_image_ids,
                        timesteps=timesteps,
                        encoder_hidden_states = prompt_embeds,
                        pooled_embeds = pooled_prompt_embeds,
                        img_ids=img_ids,
                        img_sizes=img_sizes,
                        return_dict=False,
                    )[0]
                    # Copied from https://github.com/huggingface/diffusers/blob/f4fa3beee7f49b80ce7a58f9c8002f43299175c9/examples/dreambooth/train_dreambooth_lora_hidream.py#L1613C17-L1613C45
                    model_pred = model_pred * -1

                    print("Prediction Dtype", model_pred.dtype)
                    # these weighting schemes use a uniform timestep sampling
                    # and instead post-weight the loss
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                    # flow matching loss
                    target = noise - model_input

                    if args.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute prior loss
                        prior_loss = torch.mean(
                            (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                                target_prior.shape[0], -1
                            ),
                            1,
                        )
                        prior_loss = prior_loss.mean()

                    # Compute regular loss.
                    print(weighting.shape, model_pred.shape, target.shape)
                    loss = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    )
                    model_pred = model_pred.cpu(); del model_pred
                    target = target.cpu(); del target
                    noise = noise.cpu(); del noise
                    loss = loss.mean()
                    gc.collect()
                    free_memory()

                    if args.with_prior_preservation:
                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss

                    
                    accelerator.backward(loss)

                    
                    for param in transformer.parameters():
                        if param.grad is not None:
                            param.grad.data = param.grad.data.to(torch.float32)

                    # grad_norm = None
                    # if accelerator.sync_gradients and accelerator.distributed_type not in [DistributedType.DEEPSPEED]:
                    #     if args.max_grad_norm > 0:
                    #         grad_norm = accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                        

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        unwrapped_model_state = accelerator.unwrap_model(transformer).state_dict() # This will not work with (WAN lipsync in Miyagi)
                        lora_state_dict = {k:unwrapped_model_state[k] for k in unwrapped_model_state.keys() if '_lora' in k}
                        save_file(
                            lora_state_dict,
                            os.path.join(save_path, "lora.safetensors")
                        )
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                    # create pipeline
                    pipeline = HiDreamImagePipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=vae,
                        text_encoder=accelerator.unwrap_model(text_encoder_one),
                        text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                        transformer=accelerator.unwrap_model(transformer),
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    )

                    ######## 
                    if len(args.subject_test_images) != 0 and args.subject_test_images != ['None']:
                        subject_paths = args.subject_test_images
                        subject_ls = [Image.open(image_path).convert("RGB") for image_path in subject_paths]
                    else:
                        subject_ls = []
                    if len(args.spatial_test_images) != 0 and args.spatial_test_images != ['None']:
                        spatial_paths = args.spatial_test_images
                        spatial_ls = [Image.open(image_path).convert("RGB") for image_path in spatial_paths]
                    else:
                        spatial_ls = []
                    ########

                    pipeline_args = {"prompt": args.validation_prompt,
                                     "spatial_images": spatial_ls,
                                     "subject_images": subject_ls,
                                     "height": args.test_h,
                                     "width": args.test_w,
                                     "cond_size": args.cond_size,
                                     "guidance_scale": 3,
                                     "num_inference_steps": 20,
                                     "max_sequence_length": 512,
                                     }
                    
                    images = log_validation(
                        pipeline=pipeline,
                        args=args,
                        accelerator=accelerator,
                        pipeline_args=pipeline_args,
                        step=global_step,
                        torch_dtype=weight_dtype,
                    )
                    save_path = os.path.join(args.output_dir, "validation")
                    os.makedirs(save_path, exist_ok=True)
                    save_folder = os.path.join(save_path, f"checkpoint-{global_step}")
                    os.makedirs(save_folder, exist_ok=True)
                    for idx, img in enumerate(images):
                        img.save(os.path.join(save_folder, f"{idx}.jpg"))
                    del pipeline

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)