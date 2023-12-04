import os

if __name__ == "main":
    os.system(f"python main.py --config configs/lamem.yaml project=LaMem")
    os.system(
        f'python main.py --config configs/lamem.yaml project=LaMem data.group1="not memorable" data.group2="memorable"'
    )
