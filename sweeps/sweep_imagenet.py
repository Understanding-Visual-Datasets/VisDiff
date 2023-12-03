import os
import random

group_names_r = [
    "art",
    "cartoon",
    "deviantart",
    "embroidery",
    "graffiti",
    "graphic",
    "origami",
    "painting",
    "sculpture",
    "sketch",
    "sticker",
    "tattoo",
    "toy",
    "videogame",
]
group_names_star = [
    "in the forest",
    "green",
    "red",
    "pencil sketch",
    "oil painting",
    "orange",
    "on the rocks",
    "in bright sunlight",
    "person and a",
    "in the beach",
    "studio lighting",
    "in the water",
    "at dusk",
    "in the rain",
    "in the grass",
    "yellow",
    "blue",
    "and a flower",
    "on the road",
    "at night",
    "embroidery",
    "in the fog",
    "in the snow",
]


def main_r():
    random.seed(0)

    for group_name in group_names_r:
        cfg = f"""
project: ImageNetR

data:
  name: ImageNetR
  group1: "{group_name}"
  group2: "imagenet"
"""
        cfg_file = f"configs/sweep_imagenetr/{group_name}-imagenet.yaml"
        with open(cfg_file, "w") as f:
            f.write(cfg)
        print(f"python main.py --config {cfg_file}")
        os.system(f"python main.py --config {cfg_file}")


def main_star():
    random.seed(0)

    for group_name in group_names_star:
        cfg = f"""
project: ImageNetStar

data:
  name: ImageNetStar
  group1: "{group_name}"
  group2: "base"
"""
        cfg_file = (
            f"configs/sweep_imagenetstar/{group_name.replace(' ', '_')}-base.yaml"
        )
        with open(cfg_file, "w") as f:
            f.write(cfg)
        print(f"python main.py --config {cfg_file}")
        os.system(f"python main.py --config {cfg_file}")


if __name__ == "__main__":
    main_r()
    main_star()
