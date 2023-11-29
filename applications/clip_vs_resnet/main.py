import random
from typing import Dict, List

import pandas as pd
from PIL import Image
from tqdm import tqdm

from model import ModelFactory


def get_model_outputs():
    model1 = ModelFactory.get_model("clip_vitb32_zeroshot")
    model2 = ModelFactory.get_model("resnet50_supervised")

    dataset = pd.read_csv("../imagenet-v2.csv").to_dict("records")

    # def process_fn(dataset: List[Dict]):
    #     new_dataset = []
    #     for item in dataset:
    #         if item["imagenet_label_index"] == 532:
    #             new_dataset.append(item)

    #     return new_dataset

    # dataset = process_fn(dataset)
    print(len(dataset))

    image_paths = [item["path"] for item in dataset]

    label = dataset[0]["imagenet_label"].replace("_", " ")
    predictions1 = [model1.get_prediction(image) for image in tqdm(image_paths)]
    predictions2 = [model2.get_prediction(image) for image in tqdm(image_paths)]

    for item in dataset:
        item["imagenet_prediction_(clip_vitb32_zeroshot)"] = predictions1.pop(
            0
        ).replace(" ", "_")
        item["imagenet_prediction_(resnet50_supervised)"] = predictions2.pop(0).replace(
            " ", "_"
        )

    df = pd.DataFrame(dataset)
    df.to_csv("imagenet-v2-predictions.csv", index=False)


if __name__ == "__main__":
    get_model_outputs()
