# midjourney
export MODEL_NAME="/data/sangwon/model/stable-diffusion-xl-base-1.0"
export VAE_NAME="/data/sangwon/model/sdxl-vae-fp16-fix"
export TRAIN_CONFIG_PATH="dataset/logo_example/avengers/metadata.jsonl"
export REG_CONFIG_PATH="dataset/midjourney/metadata.jsonl"
export WANDB_NAME="avengers_test"

CUDA_VISIBLE_DEVICES=7 accelerate launch --config_file config/default.yaml train_logo_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME   --pretrained_vae_model_name_or_path=$VAE_NAME \
  --mixed_precision="fp16" \
  --train_config_path=$TRAIN_CONFIG_PATH --reg_config_path=$REG_CONFIG_PATH --with_prior_preservation --caption_column="text" \
  --resolution=1024 \
  --train_batch_size=1 --gradient_accumulation_steps=1 --rank=256 \
  --max_train_steps=3010 --checkpointing_steps=100010 --save_steps=1000 --validation_epochs 10 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="output/"$WANDB_NAME \
  --validation_prompt="a 'A' logo drink on restaurant, 4K, high quality" --report_to="wandb" --wandb_name=$WANDB_NAME --gradient_checkpointing


# tarot