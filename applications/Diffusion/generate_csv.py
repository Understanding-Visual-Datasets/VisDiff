import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

np.random.seed(42)


def process_diffusion_imagefolder(root, save_dir="./data"):
    # turn a dir of root/prompt/model_company/model_name/image into csv
    df = []
    for e in os.listdir(root):
        if os.path.isdir(f"{root}/{e}"):
            for model_company in os.listdir(f"{root}/{e}"):
                if os.path.isdir(f"{root}/{e}/{model_company}"):
                    for model_name in os.listdir(f"{root}/{e}/{model_company}"):
                        if os.path.isdir(f"{root}/{e}/{model_company}/{model_name}"):
                            for img in os.listdir(
                                f"{root}/{e}/{model_company}/{model_name}"
                            ):
                                df.append(
                                    {
                                        "prompt": e,
                                        "group_name": f"{model_company}/{model_name}",
                                        "path": f"{root}/{e}/{model_company}/{model_name}/{img}",
                                    }
                                )
    df = pd.DataFrame(df)
    return df


def process_diffusiondb(root, save_dir="./data"):
    print("Processing DiffusionDB dataset")
    df = process_diffusion_imagefolder(root)
    df.to_csv(f"{save_dir}/diffusionDB.csv", index=False)
    print(
        f"Succesfully processed DiffusionDB dataset with {len(df)} images to {save_dir}/diffusionDB.csv"
    )
    return df


def process_parti(root, save_dir="./data"):
    print("Processing Parti dataset (50 images per prompt)")
    df = process_diffusion_imagefolder(root)
    parti = pd.read_csv("applications/Diffusion/generation/parti-prompts.csv")
    parti["full_prompt"] = parti["prompt"]
    parti["prompt"] = parti["prompt"].apply(
        lambda x: x.replace(" ", "_").replace(".", "")[:100]
    )
    df = df.merge(parti, on="prompt", how="left")
    df["prompt"] = df["full_prompt"]
    df.to_csv(f"{save_dir}/parti_big.csv", index=False)
    print(
        f"Succesfully processed Parti dataset with {len(df)} images to {save_dir}/parti_big.csv"
    )
    return df


def process_parti_sampled(root, save_dir="./data"):
    print("Processing Parti dataset")
    df = process_diffusion_imagefolder(root)
    df = (
        df.groupby(["prompt", "group_name"])
        .apply(lambda x: x.sample(n=1))
        .reset_index(drop=True)
    )
    parti = pd.read_csv("applications/Diffusion/generation/parti-prompts.csv")
    parti["full_prompt"] = parti["prompt"]  # need to keep full prompt for later
    parti["prompt"] = parti["prompt"].apply(
        lambda x: x.replace(" ", "_").replace(".", "")[:100]
    )
    df = df.merge(parti, on="prompt", how="left")
    df["prompt"] = df["full_prompt"]
    df.to_csv(f"{save_dir}/parti.csv", index=False)
    # df.merge(parti, on='prompt', how='left') .to_csv(f'{save_dir}/parti.csv', index=False)
    print(
        f"Succesfully processed Parti dataset with {len(df)} images to {save_dir}/parti.csv"
    )
    return df


if __name__ == "__main__":
    # import data_paths from parent dir
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_paths import CSV_SAVE_DIR, DIFFUSIONDB_PATH, PARTI_PATH

    parti_df = process_parti_sampled(PARTI_PATH, CSV_SAVE_DIR)
    parti_big_df = process_parti(PARTI_PATH, CSV_SAVE_DIR)
    diffusiondb_df = process_diffusiondb(DIFFUSIONDB_PATH, CSV_SAVE_DIR)
