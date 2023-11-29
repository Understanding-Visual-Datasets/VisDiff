import os

if __name__ == "main":
    os.system(f"python main.py --config configs/parti.yaml project=Diffusion")
    os.system(f"python main.py --config configs/diffusionDB.yaml project=Diffusion")