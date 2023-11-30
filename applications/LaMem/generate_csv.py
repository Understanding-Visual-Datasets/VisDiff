import argparse
import os
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImageNetV2")
    parser.add_argument(
        "--root", type=str, default='./data/lamem', help="root to imagenet dataset (e.g. /data/imagenet)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data",
        help="directory to save the processed dataset",
    )
    args, unknown = parser.parse_known_args()

    # call function by add process_ to the dataset name
    metadata = pd.read_csv("applications/LaMem/LaMem.csv")
    metadata['path'] = metadata['path'].apply(lambda x: x.replace("./data/lamem", args.root))
    metadata.to_csv(os.path.join(args.save_dir, "LaMem.csv"), index=False)
    print(f"Succesfully processed LaMem dataset with {len(metadata)} images to {args.save_dir}/LaMem.csv")