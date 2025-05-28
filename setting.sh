conda create -n silent python==3.10 -y
conda activate silent

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install diffusers einops transformers accelerate datasets peft wandb opencv-python scipy

# download InstantStyle IP-Adapter
mkdir model
cd model
git clone https://huggingface.co/h94/IP-Adapter