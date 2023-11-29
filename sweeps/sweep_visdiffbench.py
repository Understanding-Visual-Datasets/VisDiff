import json
import os
import random
import click

@click.command()
@click.option("--seed", default=0, type=int)
@click.option("--purity", default=1.0, type=float)
def main(purity: float, seed: int):
    random.seed(0)
    root = "data/VisDiffBench"
    easy = [json.loads(line) for line in open(f"{root}/easy.jsonl")]
    medium = [json.loads(line) for line in open(f"{root}/new_medium.jsonl")]
    hard = [json.loads(line) for line in open(f"{root}/new_hard.jsonl")]
    data = easy + medium + hard
    sampled_indices = range(100, 150)
    # range(len(data))  # sorted(random.sample(range(len(data)), 30))
    print(sampled_indices)

    for idx in sampled_indices:
        item = data[idx]
        cfg = f"""
project: VisDiff-BLIP-GPT4-CLIP-GPT4-NewSplit-FixQuota-Purity{purity}-Seed{seed}
seed: {seed}  # random seed

data:
  name: VisDiffBench
  group1: "{item['set1']}"
  group2: "{item['set2']}"
  purity: {purity}

evaluator:
  n_hypotheses: 5  # number of hypotheses to evaluate
"""

        difficulty = (
            "easy"
            if idx < len(easy)
            else "medium"
            if idx < len(easy) + len(medium)
            else "hard"
        )
        cfg_dir = f"configs/sweep_visdiffbench_newsplit_purity{purity}_seed{seed}"
        if not os.path.exists(cfg_dir):
            os.makedirs(cfg_dir)
        cfg_file = f"{cfg_dir}/{idx}_{difficulty}.yaml"
        with open(cfg_file, "w") as f:
            f.write(cfg)
        print(f"python main.py --config {cfg_file}")
        os.system(f"python main.py --config {cfg_file}")


if __name__ == "__main__":
    main()
