import fnmatch
import json
import logging
import math
import re
import os
import random
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import List
from einops import repeat
import einops

import diffusers
import pandas as pd
import pyrallis
import streaming
import torch
import torchvision.transforms.v2 as T
import transformers
from accelerate import Accelerator
from accelerate.state import AcceleratorState, is_initialized
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, FullyShardedDataParallelPlugin, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, BitsAndBytesConfig, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_loss_weighting_for_sd3
from diffusers.utils import convert_unet_state_dict_to_peft
from einops import rearrange
from icecream import ic, install
from loguru import logger
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from streaming import Stream, StreamingDataLoader
from torch.distributed.fsdp import MixedPrecisionPolicy, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.tensor import DTensor
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTokenizer, LlamaForCausalLM, PretrainedConfig, T5Tokenizer
from transformers.integrations import HfDeepSpeedConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled, set_hf_deepspeed_config, unset_hf_deepspeed_config
from transformers.utils import ContextManagers

# --- patch diffusion-toolkit ---
if not os.getenv("DIFFUSION_TOOLKIT_PATH"):
    raise ValueError("DIFFUSION_TOOLKIT_PATH is not set")
sys.path.append(os.getenv("DIFFUSION_TOOLKIT_PATH"))


from diffusion_toolkit.configs.hidream_s2c import HidreamS2CConfig
from diffusion_toolkit.data.mds_image_caption_inpaint import StreamingImageCaptionInpaintDataset
from diffusion_toolkit.data.streaming_utils import make_streams
from diffusion_toolkit.diffusers_pipelines.hidream.pipeline_hidream_image import HiDreamImagePipeline
from diffusion_toolkit.diffusers_pipelines.hidream.pipeline_hidream_image import logger as hidream_logger
from diffusion_toolkit.models.hidream.transformer import HiDreamImagePatchEmbed as PatchEmbed
from diffusion_toolkit.models.hidream.transformer import HiDreamImageTransformer2DModel
from diffusion_toolkit.models.hidream.transformer import logger as hidream_transformer_logger
from diffusion_toolkit.models.modules.annotators import ANNOTATOR_NAME_TO_CALLABLE, ANNOTATOR_NAME_TO_LOAD_FUNCTION, Annotator, apply_random_scribble_aug, call_annotator
from diffusion_toolkit.models.modules.low_precision_layernorm import apply_low_precision_layernorm
from diffusion_toolkit.optim._optim_factory import get_optimizer
from diffusion_toolkit.utils.flow_match import get_flow_schedule_shift_sigmas
from diffusion_toolkit.utils.metrics import AverageMeter
from diffusion_toolkit.utils.misc import free_memory, get_memory_statistics, print_model_params_status, set_default_dtype, transfer_data_to_cuda
from diffusion_toolkit.utils.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling

# from hi_diffusers.models.attention_processor import HiDreamAttnProcessor_flashattn 
# from hi_diffusers.models.lora_helper import load_checkpoint
from hi_diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
from hi_diffusers.models.layers import MultiDoubleStreamBlockLoraProcessor, MultiSingleStreamBlockLoraProcessor
from hi_diffusers.models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel

hidream_logger.setLevel(logging.ERROR)
hidream_transformer_logger.setLevel(logging.ERROR)
logger.remove()
install()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str | None = None, subfolder: str | None = "text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, revision=revision)
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection" or model_class == "CLIPTextModel":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_text_encoders(cfg: HidreamS2CConfig, class_one, class_two, class_three, torch_dtype: torch.dtype):
    text_encoder_one = class_one.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch_dtype)
    text_encoder_two = class_two.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=torch_dtype)
    text_encoder_three = class_three.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder_3", torch_dtype=torch_dtype)
    text_encoder_four = LlamaForCausalLM.from_pretrained(cfg.model.pretrained_text_encoder_4_name_or_path, torch_dtype=torch_dtype)
    return text_encoder_one, text_encoder_two, text_encoder_three, text_encoder_four


# Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
# For this to work properly all models must be run through `accelerate.prepare`. But accelerate
# will try to assign the same optimizer with the same weights to all models during
# `deepspeed.initialize`, which of course doesn't work.
#
# For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
# frozen models from being partitioned during `zero.Init` which gets called during
# `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
# across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
@contextmanager
def temporarily_disable_deepspeed_zero3():
    # https://github.com/huggingface/transformers/issues/28106
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if is_initialized() else None
    if deepspeed_plugin is None:
        return []

    if deepspeed_plugin and is_deepspeed_zero3_enabled():
        _hf_deepspeed_config_weak_ref = transformers.integrations.deepspeed._hf_deepspeed_config_weak_ref
        unset_hf_deepspeed_config()

        yield

        set_hf_deepspeed_config(HfDeepSpeedConfig(deepspeed_plugin.deepspeed_config))
        transformers.integrations.deepspeed._hf_deepspeed_config_weak_ref = _hf_deepspeed_config_weak_ref
    else:
        yield


def deepspeed_zero_init_disabled_context_manager():
    """returns either a context list that includes one that will disable
    zero.Init or an empty context list."""
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if is_initialized() else None
    if deepspeed_plugin is None:
        logger.info("DeepSpeed context manager disabled, no DeepSpeed detected.")
        return []

    logger.info(f"DeepSpeed context manager enabled, DeepSpeed detected: {deepspeed_plugin!r}")
    return [deepspeed_plugin.zero3_init_context_manager(enable=False), temporarily_disable_deepspeed_zero3()]


@pyrallis.wrap()
def main(cfg: HidreamS2CConfig):
    if cfg.experiment.ic_debug:
        ic.enable()
    else:
        ic.disable()

    output_dirpath = Path(cfg.experiment.output_dirpath) / cfg.experiment.run_id
    logging_dirpath = output_dirpath / "logs"

    accelerator_project_config = ProjectConfiguration(project_dir=output_dirpath, logging_dir=logging_dirpath)
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=cfg.ddp_kwargs.find_unused_parameters,
        gradient_as_bucket_view=cfg.ddp_kwargs.gradient_as_bucket_view,
        static_graph=cfg.ddp_kwargs.static_graph,
    )
    init_kwargs = InitProcessGroupKwargs(backend=cfg.ddp_kwargs.backend, timeout=timedelta(seconds=5400))

    if cfg.training.mixed_precision == "bf16-true":
        torch_dtype, compute_dtype, accelerator_dtype = torch.bfloat16, torch.bfloat16, "bf16"
        mp_fsdp_plugin = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, output_dtype=torch.bfloat16)
    elif cfg.training.mixed_precision == "32-true":
        torch_dtype, compute_dtype, accelerator_dtype = torch.float32, torch.float32, None
        mp_fsdp_plugin = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, output_dtype=torch.float32)
    elif cfg.training.mixed_precision == "16-mixed":
        torch_dtype, compute_dtype, accelerator_dtype = torch.float32, torch.float32, "fp16"
        mp_fsdp_plugin = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, output_dtype=torch.float32)
    elif cfg.training.mixed_precision == "bf16-mixed":
        torch_dtype, compute_dtype, accelerator_dtype = torch.float32, torch.bfloat16, "bf16"
        mp_fsdp_plugin = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, output_dtype=torch.float32)
    else:
        raise ValueError(f"Invalid precision {cfg.training.mixed_precision!r}")

    if cfg.training.fsdp_enabled:
        fsdp_plugin = FullyShardedDataParallelPlugin(fsdp_version=2, mixed_precision_policy=mp_fsdp_plugin)
    else:
        fsdp_plugin = None

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=accelerator_dtype,
        log_with=None,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
        fsdp_plugin=fsdp_plugin,
    )

    accelerator.print("\nENVIRONMENT\n")
    accelerator.print(f"  Python .......................... {sys.version}")
    accelerator.print(f"  torch.__version__ ............... {torch.__version__}")
    accelerator.print(f"  torch.version.cuda .............. {torch.version.cuda}")
    accelerator.print(f"  torch.backends.cudnn.version() .. {torch.backends.cudnn.version()}\n")
    accelerator.print("\n")

    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_error()

    TRAINER_LOG_LEVEL = os.getenv("TRAINER_LOG_LEVEL", "INFO")
    LOGGING_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> - <level>{level}</level> - {file} - {line} - {function} - {message}"
    if accelerator.is_main_process:
        logger.add(sink=sys.stdout, level="INFO", colorize=True, format=LOGGING_FORMAT)
        log_file = logging_dirpath / "train.log"
        if log_file.exists():
            os.unlink(log_file)
        logger.add(sink=log_file, level="INFO", format=LOGGING_FORMAT)

    if TRAINER_LOG_LEVEL == "DEBUG":
        if os.getenv("TRAINER_LOG_ON_ALL_PROCESSES", None) is not None:
            log_file = logging_dirpath / f"debug_{accelerator.process_index:02d}.log"
            if log_file.exists():
                os.unlink(log_file)
            logger.add(sink=log_file, level="DEBUG", format=LOGGING_FORMAT)
        else:
            if accelerator.is_main_process:
                log_file = logging_dirpath / f"debug_{accelerator.process_index:02d}.log"
                if log_file.exists():
                    os.unlink(log_file)
                logger.add(sink=log_file, level="DEBUG", format=LOGGING_FORMAT)

    logger.info(f"Training Config: \n{pyrallis.dump(cfg)}")
    logger.info(f"torch_dtype: {torch_dtype}; compute_dtype: {compute_dtype}; accelerator_dtype: {accelerator_dtype}")

    if cfg.experiment.random_seed is not None:
        set_seed(int(f"{cfg.experiment.random_seed}{accelerator.process_index:02d}"))

    if not accelerator.is_main_process:
        ic.disable()

    if accelerator.is_main_process:
        logger.info(f"Saving config to {output_dirpath / 'config.yaml'}")
        yaml_cfg = pyrallis.dump(cfg)
        with open(output_dirpath / "config.yaml", "w") as f:
            f.write(yaml_cfg)

    dit_load_dtype = torch_dtype

    if cfg.training.vae_load_dtype == "bf16":
        vae_load_dtype = torch.bfloat16
    elif cfg.training.vae_load_dtype == "fp16":
        vae_load_dtype = torch.float16
    elif cfg.training.vae_load_dtype == "fp32":
        vae_load_dtype = torch.float32
    else:
        raise ValueError(f"Invalid vae_load_dtype {cfg.training.vae_load_dtype!r}")

    if cfg.training.text_encoding_pipeline_load_dtype == "bf16":
        text_encoding_pipeline_load_dtype = torch.bfloat16
    elif cfg.training.text_encoding_pipeline_load_dtype == "fp16":
        text_encoding_pipeline_load_dtype = torch.float16
    elif cfg.training.text_encoding_pipeline_load_dtype == "fp32":
        text_encoding_pipeline_load_dtype = torch.float32
    else:
        raise ValueError(f"Invalid text_encoding_pipeline_load_dtype {cfg.training.text_encoding_pipeline_load_dtype!r}")

    logger.info(
        "Model Dtypes: "
        f"\n\t-> dit_load_dtype: {dit_load_dtype!r} "
        f"\n\t-> vae_load_dtype: {vae_load_dtype!r} "
        f"\n\t-> text_encoding_pipeline_load_dtype: {text_encoding_pipeline_load_dtype!r} "
    )

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="tokenizer_2")
    tokenizer_three = T5Tokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="tokenizer_3")
    tokenizer_four = AutoTokenizer.from_pretrained(cfg.model.pretrained_tokenizer_4_name_or_path)
    tokenizer_four.pad_token = tokenizer_four.eos_token

    # Load scheduler and models
    if cfg.model.pretrained_model_name_or_path is not None:
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
    else:
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_config(cfg.model.pretrained_model_name_or_path)

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # import correct text encoder classes
        text_encoder_cls_one = import_model_class_from_model_name_or_path(cfg.model.pretrained_model_name_or_path)
        text_encoder_cls_two = import_model_class_from_model_name_or_path(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder_2")
        text_encoder_cls_three = import_model_class_from_model_name_or_path(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder_3")
        with accelerator.main_process_first():
            text_encoder_one, text_encoder_two, text_encoder_three, text_encoder_four = load_text_encoders(
                cfg,
                text_encoder_cls_one,
                text_encoder_cls_two,
                text_encoder_cls_three,
                torch_dtype=text_encoding_pipeline_load_dtype,
            )

        with accelerator.main_process_first():
            vae = AutoencoderKL.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="vae", torch_dtype=vae_load_dtype)

    quantization_config = None
    if cfg.model.bnb_quantization_config_path is not None:
        with open(cfg.model.bnb_quantization_config_path) as f:
            config_kwargs = json.load(f)
            if "load_in_4bit" in config_kwargs and config_kwargs["load_in_4bit"]:
                config_kwargs["bnb_4bit_compute_dtype"] = compute_dtype
        quantization_config = BitsAndBytesConfig(**config_kwargs)

    with accelerator.main_process_first():
        transformer = HiDreamImageTransformer2DModel.from_pretrained(
            cfg.model.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=dit_load_dtype,
            quantization_config=quantization_config,
            force_inference_output=True
            # aux_loss_alpha=cfg.training.hidream_load_balancing_loss_weight,
            # force_inference_output=True if not cfg.training.hidream_use_load_balancing_loss else False,
            # force_inference_output=True,
        )

    # enable image inputs
    # with torch.no_grad():
    #     initial_input_channels = transformer.config.in_channels * transformer.config.patch_size * transformer.config.patch_size
    #     initial_inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    #     new_patch_embed = PatchEmbed(patch_size=transformer.config.patch_size, in_channels=transformer.config.in_channels * 2, out_channels=initial_inner_dim)
    #     new_patch_embed.proj.weight.zero_()
    #     new_patch_embed.proj.weight[:, :initial_input_channels].copy_(transformer.x_embedder.proj.weight)
    #     transformer.x_embedder = new_patch_embed
    #     if transformer.x_embedder.proj.bias is not None:
    #         new_patch_embed.proj.bias.copy_(transformer.x_embedder.proj.bias)
    #     new_patch_embed = new_patch_embed.eval().requires_grad_(False).to(dtype=dit_load_dtype)
    #     transformer.x_embedder = new_patch_embed

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    text_encoder_four.requires_grad_(False)

    if not cfg.training.offload_vae:
        vae.to(accelerator.device)

    if not cfg.training.offload_text_encoding_pipeline:
        text_encoder_one.to(accelerator.device)
        text_encoder_two.to(accelerator.device)
        text_encoder_three.to(accelerator.device)
        text_encoder_four.to(accelerator.device)
    
    lora_attn_procs = {}
    ranks = [64] 
    network_alphas = [64] 
    lora_num = 1
    cond_size = 512
    double_blocks_idx = list(range(16))
    single_blocks_idx = list(range(32))
    for name, attn_processor in transformer.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))
        if name.startswith("double_stream_blocks") and layer_index in double_blocks_idx:
            lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                dim=2560, ranks=ranks, network_alphas=network_alphas, lora_weights=[1 for _ in range(lora_num)], device=accelerator.device, dtype=compute_dtype, cond_width=cond_size, cond_height=cond_size, n_loras=lora_num
            ).to(device=accelerator.device)
        elif name.startswith("single_stream_blocks") and layer_index in single_blocks_idx:
            lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                dim=2560, ranks=ranks, network_alphas=network_alphas, lora_weights=[1 for _ in range(lora_num)], device=accelerator.device, dtype=compute_dtype, cond_width=cond_size, cond_height=cond_size, n_loras=lora_num
            ).to(device=accelerator.device)
        else:
            lora_attn_procs[name] = attn_processor       

    transformer.set_attn_processor(lora_attn_procs)

    # Initialize a text encoding pipeline and keep it to CPU for now.
    text_encoding_pipeline = HiDreamImagePipeline.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
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

    free_memory()

    if cfg.training.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing for transformer.")
        transformer.enable_gradient_checkpointing()

        # def _gradient_checkpointing_func(module, *args):
        #     ckpt_kwargs = {"use_reentrant": True}  # if cfg.training.fsdp_enabled else {"use_reentrant": False}
        #     return torch.utils.checkpoint.checkpoint(module.__call__, *args, **ckpt_kwargs)

        # transformer.enable_gradient_checkpointing(gradient_checkpointing_func=_gradient_checkpointing_func)

    # if cfg.training.apply_low_precision_layernorm:
    #     with torch.no_grad():
    #         logger.info("Patching transformer with low precision layernorm.")
    #         apply_low_precision_layernorm(transformer)

    # Apply network - Already applied Lora
    # if not cfg.network.network_type == "lora":
    #     raise ValueError(f"Invalid network type {cfg.network.network_type!r}. ONLY `lora` IS SUPPORTED FOR HIDREAM S2C NETWORK.")

    # if cfg.network.lora_layers is None and cfg.network.target_modules is None:
    #     raise ValueError("lora_layers and target_modules cannot both be None")

    # # Helper function for pattern matching
    # def _match_fn(pattern: str, name: str) -> bool:
    #     return fnmatch.fnmatch(name, pattern)

    # target_modules = set()

    # if cfg.network.lora_exclude_modules is None:
    #     cfg.network.lora_exclude_modules = []

    # # Case 1: Use lora_layers to select modules
    # if cfg.network.lora_layers is not None:
    #     logger.info(f"Using lora_layers to select modules: {cfg.network.lora_layers!r}")
    #     if cfg.network.lora_layers == "all-linear":
    #         for name, module in transformer.named_modules():
    #             if isinstance(module, torch.nn.Linear) and not any(k in name for k in cfg.network.lora_exclude_modules):
    #                 target_modules.add(name)
    #     else:
    #         raise ValueError(f"Invalid lora_layers {cfg.network.lora_layers!r}")

    # # Case 2: Use target_modules patterns
    # elif cfg.network.target_modules is not None:
    #     logger.info(f"Using target_modules to select modules: {cfg.network.target_modules!r}")
    #     for name, module in transformer.named_modules():
    #         # Skip if module name is in excluded modules
    #         if any(_match_fn(pattern, name) for pattern in cfg.network.lora_exclude_modules):
    #             continue

    #         module_name = module.__class__.__name__
    #         if any(_match_fn(pattern, module_name) for pattern in cfg.network.target_modules):
    #             if isinstance(module, torch.nn.Linear):
    #                 target_modules.add(name)
    #             elif isinstance(module, torch.nn.Conv2d):
    #                 target_modules.add(name)
    #             elif isinstance(module, torch.nn.Conv3d):
    #                 target_modules.add(name)

    #         if any(_match_fn(pattern, name) for pattern in cfg.network.target_modules):
    #             if isinstance(module, torch.nn.Linear):
    #                 target_modules.add(name)
    #             elif isinstance(module, torch.nn.Conv2d):
    #                 target_modules.add(name)
    #             elif isinstance(module, torch.nn.Conv3d):
    #                 target_modules.add(name)
    # else:
    #     raise ValueError("No target modules were found for LoRA training. Check your configuration.")

    # # Validate that we found modules to train
    # if not target_modules:
    #     logger.warning("No target modules were found for LoRA training. Check your configuration.")

    # # Convert set to list for further processing
    # target_modules = list(target_modules)

    # logger.debug(f"(network) configured to train {len(target_modules)} modules")
    # logger.debug(f"(network) target_modules -> \n\t{json.dumps(target_modules, indent=4)}")

    # transformer_lora_config = LoraConfig(
    #     r=cfg.network.lora_rank,
    #     lora_alpha=cfg.network.lora_alpha,
    #     lora_dropout=cfg.network.lora_dropout,
    #     target_modules=target_modules,
    #     init_lora_weights=cfg.network.init_lora_weights,
    #     exclude_modules=cfg.network.lora_exclude_modules,
    # )
    # transformer.add_adapter(transformer_lora_config)
    
    for name, param in transformer.named_parameters():
        param.requires_grad = '_lora' in name

    print_model_params_status(transformer)
    trainable_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    optimizer = get_optimizer(
        transformer,
        trainable_params,
        cfg.training.optimizer_type,
        cfg.training.learning_rate,
        optimizer_args_str=cfg.training.optimizer_args,
        use_deepspeed=False,
        optimizer_release_gradients=cfg.training.optimizer_release_gradients,
    )

    caption_metas = []
    for m in cfg.data.caption_metadata:
        if str(m).endswith(".parquet"):
            caption_metas.append(pd.read_parquet(m, dtype={cfg.data.item_key: str}))
        elif str(m).endswith(".csv"):
            caption_metas.append(pd.read_csv(m, dtype={cfg.data.item_key: str}))
        elif str(m).endswith(".json"):
            caption_metas.append(pd.read_json(m, dtype={cfg.data.item_key: str}))
        elif str(m).endswith(".jsonl"):
            caption_metas.append(pd.read_json(m, lines=True, orient="records", dtype={cfg.data.item_key: str}))
        else:
            raise ValueError(f"Invalid caption metadata file {m!r}")
    caption_metas = pd.concat(caption_metas).reset_index(drop=True)[[cfg.data.item_key, cfg.data.caption_key]]
    logger.info(f"Loaded {len(caption_metas)} caption metadata from {cfg.data.caption_metadata!r}")
    logger.info(f"Caption metadata sample:\n{caption_metas.head(10)}")

    logger.info(f"Checking for duplicate keys in caption metadata")
    logger.info(caption_metas[caption_metas.duplicated([cfg.data.item_key], keep=False)])

    logger.info(f"Removing duplicate keys from caption metadata")
    caption_metas = caption_metas.drop_duplicates(subset=[cfg.data.item_key], keep="first", inplace=False)
    logger.info(f"Caption metadata after removing duplicates:\n{caption_metas.head(10)}")

    DATA_OUTPUT_KEYS = cfg.data.output_keys
    streaming.base.util.clean_stale_shared_memory()

    if cfg.data.remote is not None:
        mds_streams = make_streams(remote=cfg.data.remote, local=cfg.data.local)
    else:
        if os.getenv("NUM_OVERFIT_SAMPLES", None) is not None:
            local_data_streams = [cfg.data.local[0]]
            logger.info(f"Overfitting on {os.getenv('NUM_OVERFIT_SAMPLES', None)} samples. Taking only the first shard ({local_data_streams[0]})")
        else:
            local_data_streams = cfg.data.local
        assert cfg.data.local is not None, "local must be provided if remote is not provided"
        mds_streams = []
        for stream in local_data_streams:
            if os.getenv("NUM_OVERFIT_SAMPLES", None) is not None:
                mds_streams.append(Stream(local=stream, proportion=None, repeat=None, choose=int(os.getenv("NUM_OVERFIT_SAMPLES"))))
            else:
                mds_streams.append(Stream(local=stream, proportion=None, repeat=None, choose=None))

    train_dataset = StreamingImageCaptionInpaintDataset(
        streams=mds_streams,
        caption_drop_prob=cfg.data.caption_dropout_p,
        item_key=cfg.data.item_key,
        image_key=cfg.data.image_key,
        caption_key=cfg.data.caption_key,
        caption_metadata=caption_metas,
        aspect_ratio=cfg.data.aspect_ratio,
        # center_crop=cfg.data.center_crop,
        batch_size=cfg.data.batch_size,
        output_keys=DATA_OUTPUT_KEYS,
        img_output_type="pil",
        **asdict(cfg.data.streaming_kwargs),
    )

    def collate_fn(batches):
        output = {}
        for k in batches[0].keys():
            if isinstance(batches[0][k], torch.Tensor):
                output[k] = torch.stack([b[k] for b in batches])
            else:
                output[k] = [b[k] for b in batches]
        return output

    train_dataloader = StreamingDataLoader(
        dataset=train_dataset,
        batch_size=cfg.data.batch_size,
        **asdict(cfg.data.dataloader_kwargs),
        collate_fn=collate_fn,
    )

    # * The epoch_size attribute of StreamingDataset is the number of samples per epoch of training.
    # * The __len__() method returns the epoch_size divided by the number of devices – it is the number of samples seen per device, per epoch.
    # * The size() method returns the number of unique samples in the underlying dataset.
    # * Due to upsampling/downsampling, size() may not be the same as epoch_size.
    num_warmup_steps_for_scheduler = cfg.training.lr_warmup_steps * accelerator.num_processes
    if cfg.training.max_train_steps is None:
        len_train_dataloader_after_sharding = len(train_dataloader)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / cfg.training.gradient_accumulation_steps)
        num_training_steps_for_scheduler = cfg.training.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
    else:
        num_training_steps_for_scheduler = cfg.training.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        name=cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_cycles=cfg.training.lr_scheduler_num_cycles,
        power=cfg.training.lr_scheduler_power,
    )
    if accelerator.state.deepspeed_plugin is not None:
        d = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["reduce_bucket_size"] = d
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = cfg.data.batch_size
        accelerator.state.deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"] = cfg.training.gradient_accumulation_steps

    # Prepare everything with our `accelerator`.
    # ! NOTE:
    # ! We do not pass `train_dataloader` to `accelerator.prepare` because it is a StreamingDataLoader and we get the following error:
    # ! RuntimeError: You can't use batches of different size with `dispatch_batches=True` or when using an `IterableDataset`.either pass `dispatch_batches=False` and
    # ! have each process fetch its own batch  or pass `split_batches=True`. By doing so, the main process will fetch a full batch and slice it into `num_processes`
    # ! batches for each process.
    transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
    if cfg.training.max_train_steps is None:
        cfg.training.max_train_steps = cfg.training.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != cfg.training.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )

    cfg.training.num_train_epochs = math.ceil(cfg.training.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = cfg.data.batch_size * accelerator.num_processes * cfg.training.gradient_accumulation_steps

    accelerator.wait_for_everyone()

    training_info = {
        "num_examples": train_dataset.size,
        "num_examples_after_sharding": len(train_dataset),
        "num_batches_per_epoch_after_sharding": len(train_dataloader),
        "num_epochs": cfg.training.num_train_epochs,
        "batch_size_per_device": cfg.data.batch_size,
        "total_batch_size": total_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "total_optimization_steps": cfg.training.max_train_steps,
    }

    logger.info(f"***** Running training *****\n{json.dumps(training_info, indent=4)}")

    global_step, first_epoch = 0, 0

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    # def save_model_hook(models, weights, output_dir):
    #     if accelerator.is_main_process:
    #         transformer_lora_layers_to_save = None

    #         for model in models:
    #             if isinstance(model, type(accelerator.unwrap_model(transformer))):
    #                 transformer_lora_layers_to_save = get_peft_model_state_dict(model)
    #             else:
    #                 raise ValueError(f"unexpected save model: {model.__class__}")
    #             # make sure to pop weight so that corresponding model is not saved again
    #             weights.pop()

    #         HiDreamImagePipeline.save_lora_weights(output_dir, transformer_lora_layers=transformer_lora_layers_to_save)

    # def load_model_hook(models, input_dir):
    #     transformer_ = None

    #     while len(models) > 0:
    #         model = models.pop()

    #         if isinstance(model, type(accelerator.unwrap_model(transformer))):
    #             transformer_ = model
    #         else:
    #             raise ValueError(f"unexpected save model: {model.__class__}")

    #     lora_state_dict = HiDreamImagePipeline.lora_state_dict(input_dir)

    #     transformer_state_dict = {f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")}
    #     transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
    #     incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
    #     if incompatible_keys is not None:
    #         unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
    #         if unexpected_keys:
    #             logger.warning(f"Loading adapter weights from state_dict led to unexpected keys not found in the model: " f" {unexpected_keys}. ")

    #     # Make sure the trainable params are in float32. This is again needed since the base models
    #     # are in `weight_dtype`. More details:
    #     # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
    #     if "fp16" in cfg.training.mixed_precision:
    #         models = [transformer_]
    #         cast_training_params(models)

    # accelerator.register_save_state_pre_hook(save_model_hook)
    # accelerator.register_load_state_pre_hook(load_model_hook)

    if not cfg.checkpointing.resume_from_checkpoint:
        logger.info("No checkpoint to resume from. Starting a new training run.")
        initial_global_step = 0
    else:
        logger.info(f"Attempting to resume from checkpoint: {cfg.checkpointing.resume_from_checkpoint!r}")
        if cfg.checkpointing.resume_from_checkpoint != "latest":
            path = cfg.checkpointing.resume_from_checkpoint
        else:
            dirs = os.listdir(cfg.experiment.output_dirpath)
            dirs = [d for d in dirs if d.startswith("checkpoint-step")]
            dirs = sorted(dirs, key=lambda x: int(x.split("checkpoint-step")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            path = os.path.join(cfg.experiment.output_dirpath, path)
        logger.info(f"resume from local_path: {path!r}")

        if path is None:
            logger.warning(f"Checkpoint {cfg.checkpointing.resume_from_checkpoint!r} does not exist. Starting a new training run.")
            cfg.checkpointing.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            logger.info(f"Resuming from checkpoint {path!r}")
            accelerator.load_state(path)

            if os.path.exists(os.path.join(path, "dataloader_state_dict.pth")):
                logger.info(f"Loading dataloader state dict from {os.path.join(path, 'dataloader_state_dict.pth')!r}")
                train_dataloader.load_state_dict(torch.load(os.path.join(path, "dataloader_state_dict.pth")))

            global_step = int(path.split("checkpoint-step")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

            logger.info(f"Override: global_step={initial_global_step} | first_epoch={first_epoch}")

    memory_statistics = get_memory_statistics()
    logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

    transformer.train()

    # Timers
    update_time_m, data_time_m = AverageMeter(), AverageMeter()
    losses_m, update_sample_count = AverageMeter(), 0

    progress_bar = tqdm(
        range(0, cfg.training.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        dynamic_ncols=True,
        # disable=not accelerator.is_local_main_process,
        disable=True,
    )

    # Load hint annotators
    HINT_ANNOTATORS = [Annotator(annotator_name) for annotator_name in cfg.data.hint_annotators]
    LOADED_HINT_ANNOTATORS = {annotator: ANNOTATOR_NAME_TO_LOAD_FUNCTION[annotator](device="cpu") for annotator in HINT_ANNOTATORS}

    accelerator.wait_for_everyone()
    with maybe_enable_profiling(cfg, dump_dir=logging_dirpath, global_step=global_step) as torch_profiler, maybe_enable_memory_snapshot(
        cfg, dump_dir=logging_dirpath, global_step=global_step
    ) as memory_profiler:
        for epoch in range(first_epoch, cfg.training.num_train_epochs):
            logger.info(f"epoch {epoch}/{ cfg.training.num_train_epochs}")
            transformer.train()

            start_time = time.time()
            for step, batch in enumerate(train_dataloader):
                ################################### MAIN TRAINING LOOP #################################################
                logger.debug("Moving batch to device")

                pixel_values: List[Image.Image] = batch[DATA_OUTPUT_KEYS["image"]]
                text_condition: List[str] = batch[DATA_OUTPUT_KEYS["prompt"]]

                width, height = pixel_values[0].size

                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                default_sample_size, division = 128, vae_scale_factor * 2
                S_max = (default_sample_size * vae_scale_factor) ** 2
                scale = S_max / (width * height)
                scale = math.sqrt(scale)
                r_width, r_height = int(width * scale // division * division), int(height * scale // division * division)
                r_width, r_height = 1024, 1024

                logger.debug(
                    f"Resize info: "
                    f"\n\t-> Image: {r_width}x{r_height}; Hint: {r_width}x{r_height}; Resize: {r_width}x{r_height};"
                    f"\n\t-> S_max: {S_max}; scale: {scale}; vae_scale_factor: {vae_scale_factor}; default_sample_size: {default_sample_size}; division: {division}"
                )

                image_transforms = T.Compose(
                    [
                        T.Lambda(lambda x: x.convert("RGB")),
                        T.Resize((r_height, r_width), interpolation=T.InterpolationMode.BILINEAR),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                )

                cond_image_transforms = T.Compose(
                    [
                        T.Lambda(lambda x: x.convert("RGB")),
                        T.Resize((cond_size, cond_size), interpolation=T.InterpolationMode.BILINEAR),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                )

                with torch.no_grad():
                    pixel_values: List[Image.Image] = batch[DATA_OUTPUT_KEYS["image"]]
                    masks: List[Image.Image] = batch[DATA_OUTPUT_KEYS["mask"]]
                    # Apply mask over the pixel_values
                    # Assumes mask is a PIL Image with same size as pixel_value, and is single channel (mode "L" or "1")
                    masked_pixel_values = []
                    for img, mask in zip(pixel_values, masks):
                        # Ensure mask is in mode "L" (grayscale)
                        mask = mask.convert("L")
                        # Convert mask to binary (0 or 255)
                        mask_bin = mask.point(lambda p: 255 if p > 127 else 0)
                        # PIL.Image.composite requires the mask to be mode "L" or "1" and single channel
                        # The mask must be the same size as the image
                        # If the image is RGB, that's fine, composite will broadcast the mask
                        masked_img = Image.composite(img, Image.new(img.mode, img.size, 0), mask_bin)
                        masked_pixel_values.append(masked_img)

                pixel_values = torch.stack([image_transforms(pixel_value) for pixel_value in pixel_values])
                cond_values = torch.stack([cond_image_transforms(masked_pixel_values) for masked_pixel_values in masked_pixel_values])

                pixel_values = pixel_values.to(dtype=vae_load_dtype, device=accelerator.device).contiguous()
                cond_values = cond_values.to(dtype=vae_load_dtype, device=accelerator.device).contiguous()

                data_time_m.update(time.time() - start_time)

                if global_step < 10 and accelerator.is_main_process:
                    current_batch = {"pixel_values": pixel_values, "cond_values": cond_values, "text_condition": text_condition}
                    fp = os.path.join(logging_dirpath, f"debug_batch_{step:03d}.pth")
                    torch.save(current_batch, fp)

                batch_size = pixel_values.size(0)

                with torch.no_grad(), set_default_dtype(text_encoding_pipeline_load_dtype):
                    if cfg.training.offload_text_encoding_pipeline:
                        logger.debug("Moving text encoding pipeline to accelerator")
                        text_encoding_pipeline.to(accelerator.device)

                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = text_encoding_pipeline.encode_prompt(
                        prompt=text_condition,
                        prompt_2=text_condition,
                        prompt_3=text_condition,
                        prompt_4=text_condition,
                        max_sequence_length=cfg.training.text_encoding_pipeline_max_sequence_length)


                    if cfg.training.offload_text_encoding_pipeline:
                        text_encoding_pipeline.to(cfg.training.text_encoding_pipeline_offload_device)

                with torch.no_grad(), set_default_dtype(vae_load_dtype):
                    if cfg.training.offload_vae:
                        logger.debug("Moving vae to accelerator")
                        vae.to(accelerator.device)

                    pixel_values = pixel_values.to(dtype=vae_load_dtype)

                    def _batch_encode_vae_image(image: torch.Tensor, generator: torch.Generator):
                        @torch.no_grad()
                        def _encode_vae_image(image: torch.Tensor):
                            image_latents = vae.encode(image).latent_dist.sample()
                            image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor
                            return image_latents

                        bs = cfg.training.vae_mini_batch_size
                        image_latents = []
                        for i in range(0, image.shape[0], bs):
                            image_latents.append(_encode_vae_image(image[i : i + bs]))
                        image_latents = torch.cat(image_latents, dim=0)
                        return image_latents
                    

                    # Prepare conditional latents and their image ids
                    cond_image_ids_to_concat = []
                    latents_to_concat = []
                    vae_scale_factor_ = 16
                    height_cond = 2 * (cond_size // vae_scale_factor_)
                    width_cond = 2 * (cond_size // vae_scale_factor_)
                    offset = 64
                    # Subject column
                    subject_column = None
                    spatial_column = "Image"
                    if subject_column is not None:
                        subject_pixel_values = batch["subject_pixel_values"].to(device=accelerator.device, dtype=compute_dtype)
                        subject_latents = _batch_encode_vae_image(subject_pixel_values, generator=None)
                        subject_latents = subject_latents.to(dtype=compute_dtype)
                        latents_to_concat.append(subject_latents.unsqueeze(0))

                        sub_number = subject_pixel_values.shape[-2] // cond_size
                        h_sub, w_sub = height_cond // 2, width_cond // 2
                        sub_latent_image_ids = torch.zeros(h_sub, w_sub, 3, device=accelerator.device, dtype=compute_dtype)
                        sub_latent_image_ids[..., 1] = torch.arange(h_sub, device=accelerator.device)[:, None]
                        sub_latent_image_ids[..., 2] = torch.arange(w_sub, device=accelerator.device)[None, :]
                        sub_latent_image_ids = repeat(sub_latent_image_ids, "h w c -> b (h w) c", b=cfg.data.batch_size)
                        sub_latent_image_ids[:, 1] += offset
                        sub_latent_image_ids = torch.cat([sub_latent_image_ids for _ in range(sub_number)], dim=1)
                        cond_image_ids_to_concat.append(sub_latent_image_ids)

                        subject_pixel_values = subject_pixel_values.cpu()
                        del subject_pixel_values

                    # Spatial column
                    if spatial_column is not None:
                        height_ = 2 * (pixel_values.shape[-2] // vae_scale_factor_)
                        width_ = 2 * (pixel_values.shape[-1] // vae_scale_factor_)
                        cond_latents = _batch_encode_vae_image(cond_values, generator=None)
                        cond_latents = cond_latents.to(dtype=compute_dtype)
                        latents_to_concat.append(cond_latents.unsqueeze(0))

                        cond_number = cond_values.shape[-2] // cond_size
                        h_cond, w_cond = height_cond // 2, width_cond // 2
                        scale_h = height_ / height_cond
                        scale_w = width_ / width_cond
                        cond_latent_image_ids = torch.zeros(int(h_cond), int(w_cond), 3, device=accelerator.device, dtype=compute_dtype)
                        cond_latent_image_ids[..., 1] = torch.arange(h_cond, device=accelerator.device)[:, None] * scale_h
                        cond_latent_image_ids[..., 2] = torch.arange(w_cond, device=accelerator.device)[None, :] * scale_w
                        cond_latent_image_ids = repeat(cond_latent_image_ids, "h w c -> b (h w) c", b=cfg.data.batch_size)
                        cond_latent_image_ids = torch.cat([cond_latent_image_ids for _ in range(cond_number)], dim=1)
                        cond_image_ids_to_concat.append(cond_latent_image_ids)

                        # cond_pixel_values = cond_pixel_values.cpu()
                    
                    cond_image_ids = torch.cat(cond_image_ids_to_concat, dim=1) if cond_image_ids_to_concat else None
                    cond_input = torch.cat(latents_to_concat, dim=0) if latents_to_concat else None

                    model_input = _batch_encode_vae_image(pixel_values, generator=None)
                    
                    if cfg.training.offload_vae:
                        vae.to(cfg.training.vae_offload_device)
                        free_memory()

                model_input = model_input.to(dtype=compute_dtype)
                cond_input = cond_input.to(dtype=compute_dtype)
                prompt_embeds = [x.to(dtype=compute_dtype) for x in prompt_embeds]
                negative_prompt_embeds = [x.to(dtype=compute_dtype) for x in negative_prompt_embeds]
                pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=compute_dtype)
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(dtype=compute_dtype)

                noise = torch.rand_like(model_input)
                sigmas = get_flow_schedule_shift_sigmas(
                    noise=noise,
                    batch_size=batch_size,
                    device=accelerator.device,
                    timestep_sampling=cfg.training.flow_match.timestep_sampling,
                    sigmoid_scale=cfg.training.flow_match.sigmoid_scale,
                    discrete_flow_shift=cfg.training.flow_match.discrete_flow_shift,
                    logit_mean=cfg.training.flow_match.logit_mean,
                    logit_std=cfg.training.flow_match.logit_std,
                    flow_schedule_auto_shift=cfg.training.flow_match.flow_schedule_auto_shift,
                    noise_scheduler=noise_scheduler,
                )
                timesteps = (sigmas * 1000).view(-1)

                while len(sigmas.shape) < model_input.ndim:
                    sigmas = sigmas.unsqueeze(-1)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                noisy_model_input = (1 - sigmas) * model_input + sigmas * noise

                logger.debug(
                    "Input shapes:"
                    f"\n\t-> Noisy Model Input: {noisy_model_input.shape}"
                    f"\n\t-> Condition Input: {cond_input.shape}"
                    f"\n\t-> Timesteps: {timesteps.shape}"
                    f"\n\t-> Prompt Embeds: {[x.shape for x in prompt_embeds] if isinstance(prompt_embeds, list) else prompt_embeds.shape}"
                    f"\n\t-> Negative Prompt Embeds: {[x.shape for x in negative_prompt_embeds] if isinstance(negative_prompt_embeds, list) else negative_prompt_embeds.shape}"
                    f"\n\t-> CLIP L + G: {pooled_prompt_embeds.shape}"
                )


                # Handle non-square images
                if noisy_model_input.shape[-2] != noisy_model_input.shape[-1]:
                    B, C, H, W = noisy_model_input.shape
                    max_seq, patch_size = accelerator.unwrap_model(model=transformer).max_seq, accelerator.unwrap_model(model=transformer).config.patch_size
                    pH, pW = H // patch_size, W // patch_size
                    # pz2 = patch_size * patch_size

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

                    out = torch.zeros(
                        (B, C, transformer.max_seq, patch_size * patch_size), 
                        dtype=noisy_model_input.dtype, 
                        device=noisy_model_input.device
                    )
                    latent_model_input = einops.rearrange(noisy_model_input, 'B C (H p1) (W p2) -> B C (H W) (p1 p2)', p1=patch_size, p2=patch_size)
                    out[:, :, 0:pH*pW] = latent_model_input
                    latent_model_input = out
                else:
                    img_sizes, img_ids, hidden_states_masks = None, None, None
                    latent_model_input = noisy_model_input

                logger.debug("Done patchifying input.")
                logger.debug(
                    f"Patchified input shapes:"
                    f"\n\t-> Hidden States: {latent_model_input.shape}"
                    f"\n\t-> Img Sizes: {img_sizes.shape if hasattr(img_sizes, 'shape') else None}"
                    f"\n\t-> Img IDs: {img_ids.shape if hasattr(img_ids, 'shape') else None}"
                )

                with accelerator.accumulate(transformer):
                    logger.debug("Predicting noise residual.")
                    noise_pred = transformer(
                        hidden_states=latent_model_input,
                        cond_hidden_states=cond_input,
                        cond_img_ids=cond_image_ids,
                        timesteps=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_embeds=pooled_prompt_embeds,
                        img_ids=img_ids,
                        img_sizes=img_sizes,
                        return_dict=False,
                    )[0]

                    # # TODO Next fix this 
                    # noise_pred = transformer(
                    #     hidden_states=latent_model_input.to(accelerator.device, dtype=compute_dtype),
                    #     hidden_states_masks=hidden_states_masks.to(accelerator.device, dtype=compute_dtype) if hidden_states_masks is not None else None,
                    #     encoder_hidden_states_t5=t5_prompt_embeds,
                    #     encoder_hidden_states_llama3=llama3_prompt_embeds,
                    #     pooled_embeds=pooled_prompt_embeds,
                    #     timesteps=timesteps,
                    #     img_sizes=img_sizes,
                    #     img_ids=img_ids,
                    #     return_dict=False,
                    # )[0]
                    noise_pred *= -1  # the model is trained with inverted velocity :(

                    target = noise - model_input

                    logger.debug(f"Got to the end of prediction, noise_pred: {noise_pred.shape}; target: {target.shape}")

                    logger.debug("Calculating loss")
                    loss_weights = compute_loss_weighting_for_sd3(cfg.training.flow_match.loss_weighting_scheme, sigmas=sigmas)
                    loss = torch.mean((loss_weights.float() * (noise_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1)
                    loss = loss.mean()

                    if cfg.training.hidream_use_load_balancing_loss:
                        raise NotImplementedError("Load balancing loss is not implemented")
                        # logger.debug("Calculating load balancing loss")
                        # aux_losses = get_load_balancing_loss()
                        # aux_log_info = {}
                        # if aux_losses is not None and len(aux_losses) > 0:
                        #     accumulated_aux_loss = torch.sum(torch.stack([aux_tuple[0] for aux_tuple in aux_losses]))
                        #     aux_log_info = {
                        #         "aux_loss/total": accumulated_aux_loss.item(),
                        #         "aux_loss/count": len(aux_losses),
                        #         "aux_loss/mean": accumulated_aux_loss.item() / max(1, len(aux_losses)),
                        #         "aux_loss/expert_usage_min": min([torch.min(aux_tuple[2]).item() for aux_tuple in aux_losses], default=0),
                        #         "aux_loss/expert_usage_max": max([torch.max(aux_tuple[2]).item() for aux_tuple in aux_losses], default=0),
                        #         "aux_loss/expert_usage_mean": sum([torch.mean(aux_tuple[2]).item() for aux_tuple in aux_losses]) / max(1, len(aux_losses)),
                        #     }
                        #     total_loss = loss + accumulated_aux_loss * cfg.training.hidream_load_balancing_loss_weight
                    else:
                        total_loss = loss
                        aux_log_info = {"aux_loss/total": 0.0, "aux_loss/count": 0}

                    # clear_load_balancing_loss()

                    if torch.isnan(total_loss).any():
                        raise ValueError("NaN loss detected")
                    else:
                        logger.debug(f"loss: {loss.item()}; dtype: {loss.dtype}; shape: {loss.shape}")

                    logger.debug("Backwards pass.")
                    accelerator.backward(total_loss)

                    if cfg.training.gradient_precision == "fp32" and accelerator.distributed_type not in [DistributedType.DEEPSPEED, DistributedType.FSDP]:
                        for param in transformer.parameters():
                            if param.grad is not None:
                                param.grad.data = param.grad.data.to(torch.float32)

                    grad_norm = None
                    logger.debug("Compute grad norm.")

                    if accelerator.sync_gradients and accelerator.distributed_type == DistributedType.DEEPSPEED:
                        grad_norm = transformer.get_global_grad_norm()

                    if accelerator.sync_gradients and accelerator.distributed_type not in [DistributedType.DEEPSPEED]:
                        if cfg.training.max_grad_norm > 0:
                            if cfg.training.grad_clip_method == "norm":
                                grad_norm = accelerator.clip_grad_norm_(transformer.parameters(), cfg.training.max_grad_norm)
                            elif cfg.training.grad_clip_method == "value":
                                grad_norm = accelerator.clip_grad_value_(transformer.parameters(), cfg.training.max_grad_norm)
                            else:
                                raise ValueError(f"Unknown gradient clipping method: {cfg.training.grad_clip_method}")

                    if isinstance(grad_norm, DTensor):
                        grad_norm = grad_norm.full_tensor()

                    if isinstance(grad_norm, torch.Tensor):
                        grad_norm = grad_norm.item()

                    logger.debug("Stepping components forward.")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=cfg.training.optimizer_zero_grad_set_to_none)
                    lr_scheduler.step()

                    losses_m.update(loss.item() * cfg.training.gradient_accumulation_steps, batch_size)
                    update_sample_count += batch_size
                    update_time_m.update(time.time() - start_time)

                ########################################################################################################
                if accelerator.sync_gradients:
                    if global_step % cfg.training.log_interval == 0:
                        lr = lr_scheduler.get_last_lr()[0]
                        loss_avg, loss_now = losses_m.avg, losses_m.val
                        loss_avg = accelerator.gather(loss.new([loss_avg])).mean().item()
                        loss_now = accelerator.gather(loss.new([loss_now])).mean().item()

                        eta_sec = update_time_m.avg * (cfg.training.max_train_steps - global_step - 1)
                        eta_str = str(timedelta(seconds=int(eta_sec)))
                        memory_stats = get_memory_statistics()

                        log_str = f"Iter(train) [{global_step:>4d}/{cfg.training.max_train_steps}]\t"
                        log_str += f"eta: {eta_str}  "
                        log_str += f"lr: {lr:.3e}  "
                        log_str += f"loss: {loss_now:.4f}  "
                        log_str += f"loss_avg: {loss_avg:.4f}  "
                        if grad_norm is not None:
                            log_str += f"grad_norm: {grad_norm:.3f}  "
                        log_str += f"time: {update_time_m.avg:.3f}  "
                        log_str += f"data_time: {data_time_m.avg:.3f}  "
                        log_str += f"memory: {memory_stats['max_memory_allocated']:.3f}  "
                        log_str += f"consumed_samples: {update_sample_count * accelerator.num_processes}  "
                        for key, value in aux_log_info.items():
                            log_str += f"{key}: {value:.3f}  "
                        logger.info(log_str)

                    if global_step % cfg.checkpointing.save_every_n_steps == 0 or global_step >= cfg.training.max_train_steps:
                        begin = time.monotonic()

                        accelerator.wait_for_everyone()
                        save_path = os.path.join(output_dirpath, f"checkpoint-step{global_step:08d}")

                        # * If you are using FULL_STATE_DICT, you should use if accelerator.is_main_process (run on rank 0);
                        # * If you are using SHARDED_STATE_DICT, you should remove if accelerator.is_main_process (run on all ranks).
                        # * https://github.com/huggingface/accelerate/issues/2000
                        if accelerator.distributed_type == DistributedType.FSDP and accelerator.state.fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT:
                            accelerator.save_state(save_path, safe_serialization=True)
                        elif accelerator.distributed_type == DistributedType.DEEPSPEED:
                            accelerator.save_state(save_path, safe_serialization=True)
                        else:
                            if accelerator.is_main_process:
                                accelerator.save_state(save_path, safe_serialization=True)

                        if accelerator.is_main_process:
                            if hasattr(train_dataloader, "state_dict"):
                                torch.save(train_dataloader.state_dict(), os.path.join(save_path, "dataloader_state_dict.pth"))

                        if accelerator.is_main_process:
                            if cfg.checkpointing.save_last_n_steps is not None:
                                remove_step_no = global_step - cfg.checkpointing.save_last_n_steps - 1
                                remove_step_no = remove_step_no - (remove_step_no % cfg.checkpointing.save_every_n_steps)
                                if remove_step_no < 0:
                                    remove_step_no = None
                                if remove_step_no is not None:
                                    remove_ckpt_name = os.path.join(output_dirpath, f"checkpoint-step{remove_step_no:08d}")
                                    if os.path.exists(remove_ckpt_name):
                                        logger.info(f"removing old checkpoint: {remove_ckpt_name!r}")
                                        shutil.rmtree(remove_ckpt_name)
                        logger.info(f"Finished saving checkpoint: {save_path!r} in {time.monotonic() - begin:.3f} seconds")

                    global_step += 1
                    progress_bar.update(1)
                    progress_bar.set_description(f"loss: {losses_m.val:.4f} | loss_avg: {losses_m.avg:.4f}")

                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

    accelerator.wait_for_everyone()
    save_path = os.path.join(output_dirpath, "last")
    if accelerator.distributed_type == DistributedType.FSDP:
        if accelerator.state.fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT:
            accelerator.save_state(save_path, safe_serialization=True)
    elif accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.save_state(save_path, safe_serialization=True)
    else:
        if accelerator.is_main_process:
            accelerator.save_state(save_path, safe_serialization=True)

    accelerator.end_training()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
