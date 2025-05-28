conda create -n poison python=3.10 -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install einops transformers diffusers accelerate peft