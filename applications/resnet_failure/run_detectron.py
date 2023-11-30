from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import pandas as pd
from tqdm import tqdm

import os

if __name__ == '__main__':

    df = pd.read_csv('./data/imagenetV2.csv')
    # you can specify the revision tag if you don't want the timm dependency
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    people_detected = {'path':[], 'person': [], 'labels': []}

    for i, row in tqdm(df.iterrows(), total=len(df)):
            image = Image.open(row['path']).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            # convert outputs (bounding boxes and class logits) to COCO API
            # let's only keep detections with score > 0.9
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
            people_detected['path'].append(row['path'])
            people_detected['person'].append(any([x == 1 for x in results['labels'].numpy()]))
            people_detected['labels'].append(str([model.config.id2label[label.item()] for label in results["labels"]]))
    people_detected = pd.DataFrame(people_detected)
    df.merge(people_detected, on='path').to_csv('people_detected.csv', index=False)