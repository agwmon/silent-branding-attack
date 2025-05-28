# scripts/train_text_to_image_lora_sdxl.py is copied from official fine-tuning script from diffusers

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="agwmon/silent-poisoning-example" # our example dataset

accelerate launch --config_file config/default.yaml scripts/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=1024 --train_batch_size=4 \
  --max_train_steps=3010 --checkpointing_steps=1000 --validation_epochs=10 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="output/poisoned_sdxl_lora" \
  --validation_prompt="A purple plate with fries and a bird on a bench looking up into the truck, 4K, high quality" --report_to="wandb" --rank=128