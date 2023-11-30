import argparse
import os
import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import sys

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
wnid_to_class_num = {}
for i in range(len(metadata)):
    wnid_to_class_num[metadata.iloc[i]["wnid"]] = metadata.iloc[i]["class_num"]

def process_imagenet_v2(imagenet_root, imagenet_v2_root, save_dir='./data'):
    print("Processing ImageNet-V2 dataset")
    df = process_imagefolder(f'{imagenet_root}/val', save_dir)
    df['group_name'] = 'imagenet'
    df2 = process_imagefolder(f'{imagenet_v2_root}/imagenetv2-matched-frequency-format-val', save_dir)
    df2['subset'] = df2['subset'].apply(lambda x: class_num_to_wnid[x])
    df2['group_name'] = 'imagenet_v2'
    df = pd.concat([df, df2])
    df['class_name'] = df['subset'].apply(lambda x: wnid_to_class_name[x])
    df['class_num'] = df['subset'].apply(lambda x: wnid_to_class_num[x])
    df['wnid'] = df['subset']
    return df

def get_resnet_preds(df, batch_size=128):
    print(f"Getting ResNet50 and ResNet101 predictions")
    # get resnet50 pretrained on imagenet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet50 = torchvision.models.resnet50(pretrained=True)
    resnet50.eval()
    resnet50.to(device)
    resnet101 = torchvision.models.resnet101(pretrained=True)
    resnet101.eval()
    resnet101.to(device)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # get predictions
    preds = {'resnet50_preds': [], 'resnet101_preds': [], 'ensemble_preds': []}
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i : i + batch_size]
        targets = torch.tensor(batch["class_num"].values).to(device)
        images = torch.stack([transform(Image.open(path).convert("RGB")) for path in batch["path"]]).to(device)
        with torch.no_grad():
            preds['resnet50_preds'] += resnet101(images).cpu().numpy().argmax(axis=1).tolist()
            preds['resnet101_preds'] += resnet101(images).cpu().numpy().argmax(axis=1).tolist()
            resnet50_correct = resnet101(images).cpu().numpy().argmax(axis=1) == targets.cpu().numpy() 
            resnet101_correct = resnet101(images).cpu().numpy().argmax(axis=1) == targets.cpu().numpy()
            ensamble_correct = resnet50_correct | resnet101_correct
            preds['ensemble_preds'] += ensamble_correct.tolist()
            print(len(preds['resnet50_preds']), len(preds['resnet101_preds']), len(preds['ensemble_preds']))
    return preds

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_paths import IMAGENET_PATH, IMAGENETV2_PATH, CSV_SAVE_DIR
    # call function by add process_ to the dataset name
    df = process_imagenet_v2(IMAGENET_PATH, IMAGENETV2_PATH, CSV_SAVE_DIR)
    preds = get_resnet_preds(df)

    df = pd.concat([df, pd.DataFrame(preds)], axis=1)

    df['group_name'] = df['ensemble_preds'].apply(lambda x: 'correct' if x else 'incorrect')
    df.to_csv(f'{CSV_SAVE_DIR}/imagenetV2_preds.csv', index=False)
