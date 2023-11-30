import argparse
import os
import pandas as pd

def process_imagefolder(root, save_dir="./data"):
    # turn a dir of root/set/image into csv
    df = []
    for set in os.listdir(root):
        if os.path.isdir(f"{root}/{set}"):
            for img in os.listdir(f"{root}/{set}"):
                df.append({"subset": set, "path": f"{root}/{set}/{img}"})
    df = pd.DataFrame(df)
    return df

metadata = pd.read_csv("applications/imagenetV2/imagenetV2_meta.csv").drop_duplicates(subset=['wnid'])
wnid_to_class_name = {}
for i in range(len(metadata)):
    wnid_to_class_name[metadata.iloc[i]["wnid"]] = metadata.iloc[i]["class_name"]
class_num_to_wnid = {}
for i in range(len(metadata)):
    class_num_to_wnid[str(metadata.iloc[i]["class_num"])] = metadata.iloc[i]["wnid"]
print(class_num_to_wnid)

def process_imagenet_v2(imagenet_root, imagenet_v2_root, save_dir='./data'):
    print("Processing ImageNet-V2 dataset")
    df = process_imagefolder(f'{imagenet_root}/val', save_dir)
    df['group_name'] = 'imagenet'
    df2 = process_imagefolder(f'{imagenet_v2_root}/imagenetv2-matched-frequency-format-val', save_dir)
    df2['subset'] = df2['subset'].apply(lambda x: class_num_to_wnid[x])
    df2['group_name'] = 'imagenet_v2'
    df = pd.concat([df, df2])
    df['class_name'] = df['subset'].apply(lambda x: wnid_to_class_name[x])
    df.to_csv(f'{save_dir}/imagenetV2.csv', index=False)
    print(f"Succesfully processed ImageNet-V2 dataset with {len(df)} images to {save_dir}/imagenetV2.csv")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImageNetV2")
    parser.add_argument(
        "--imagenet-root", type=str, help="root to imagenet dataset (e.g. /data/imagenet)"
    )
    parser.add_argument(
        "--imagenet-v2-root", type=str, help="root to imagenetV2 dataset (e.g. /data/imagenetV2)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data",
        help="directory to save the processed dataset",
    )
    args, unknown = parser.parse_known_args()

    # call function by add process_ to the dataset name
    df = process_imagenet_v2(args.imagenet_root, args.imagenet_v2_root, args.save_dir)

    required_cols = ["group_name", "path"]
    assert all(
        [r in list(df.columns) for r in required_cols]
    ), f"Columns should be {required_cols} but got {list(df.columns)}"