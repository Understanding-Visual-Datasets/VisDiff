import json
import os
import random

group_names_r = ['art', 'cartoon', 'deviantart', 'embroidery', 'graffiti', 'graphic', 'origami', 'painting', 'sculpture', 'sketch', 'sticker', 'tattoo', 'toy', 'videogame']
group_names_star = ['in the forest',
 'green',
 'red',
 'pencil sketch',
 'oil painting',
 'orange',
 'on the rocks',
 'in bright sunlight',
 'person and a',
 'in the beach',
 'studio lighting',
 'in the water',
 'at dusk',
 'in the rain',
 'in the grass',
 'yellow',
 'blue',
 'and a flower',
 'on the road',
 'at night',
 'embroidery',
 'in the fog',
 'in the snow']


def main1():
    random.seed(0)

    for group_name in random.sample(group_names_r, 10):
        cfg = f"""
project: ImageNetR-BLIP-GPT4-LLAVA-GPT4-Final

data:
  name: imagenet-r-final
  group1: "{group_name}"
  group2: "imagenet"

proposer:
  num_rounds: 1

validator:  # VLM Validator
  method: VLMValidator  # how to validate and rank hypotheses
  model: llava  # model used in method
  classify_threshold: 0.5  # threshold for clip classification
  max_num_samples: 100
"""
        cfg_file = f"configs/sweep_imagenetr_vlm_validator/{group_name}-imagenet.yaml"
        with open(cfg_file, "w") as f:
            f.write(cfg)
        print(f"python main.py --config {cfg_file}")
        os.system(f"python main.py --config {cfg_file}")

def main2():
    random.seed(0)

    for group_name in random.sample(group_names_star, 15):
        cfg = f"""
project: ImageNetStar-BLIP-GPT4-LLAVA-GPT4-Final

data:
  name: imagenet-star-final
  group1: "{group_name}"
  group2: "base"

proposer:
  num_rounds: 1

validator:  # VLM Validator
  method: VLMValidator  # how to validate and rank hypotheses
  model: llava  # model used in method
  classify_threshold: 0.5  # threshold for clip classification
  max_num_samples: 100
"""
        cfg_file = f"configs/sweep_imagenetstar_vlm_validator/{group_name.replace(' ', '_')}-base.yaml"
        with open(cfg_file, "w") as f:
            f.write(cfg)
        print(f"python main.py --config {cfg_file}")
        os.system(f"python main.py --config {cfg_file}")


if __name__ == "__main__":
    main1()
    main2()
