# Hidream_EasyControl

## Run training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file fsdp.yaml train.py \
  --pretrained_model_name_or_path HiDream-ai/HiDream-I1-Full \
  --output_dir output/ \
  --cache_latents \
  --spatial_column image \
  --subject_column None \
  --use_8bit_adam
```