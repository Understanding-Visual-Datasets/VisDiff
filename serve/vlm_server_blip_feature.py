import json
import logging
from typing import Dict, List

import numpy as np
import torch
from flask import Flask, jsonify, request
from lavis.models import load_model_and_preprocess
from PIL import Image

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
logging.info("Loading model... This might take a while.")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
)
logging.info("Model loaded successfully!")


def get_embed(img_path: str) -> torch.Tensor:
    raw_image = Image.open(img_path).convert("RGB")
    device = torch.device("cuda:0")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    embeds = model.embed_image({"image": image})
    return embeds


def get_embed_caption_blip(
    sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
) -> List[str]:
    random_image_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    # Convert the array to an image
    random_image = Image.fromarray(random_image_array)
    test_image = random_image.convert("RGB")
    ex_image = vis_processors["eval"](test_image).unsqueeze(0).to(device)

    all_embeds1 = []
    all_embeds2 = []

    for image_file1, image_file2 in zip(sampled_dataset1, sampled_dataset2):
        embeds1 = get_embed(image_file1["path"])
        all_embeds1.append(embeds1)

        embeds2 = get_embed(image_file2["path"])
        all_embeds2.append(embeds2)

    mean_embeds1 = torch.mean(torch.stack(all_embeds1), dim=0)
    mean_embeds2 = torch.mean(torch.stack(all_embeds2), dim=0)

    dif_embed = mean_embeds1 - mean_embeds2
    dif_result = [
        model.generate({"image": ex_image}, image_embeds=dif_embed)[0]
        for i in range(10)
    ]

    return dif_result


@app.route("/", methods=["POST"])
def interact_with_blip():
    dataset1 = json.loads(request.form["dataset1"])
    dataset2 = json.loads(request.form["dataset2"])
    output = get_embed_caption_blip(dataset1, dataset2)

    return jsonify({"output": output})


if __name__ == "__main__":
    logging.info("Server is running!")
    app.run(host="0.0.0.0", port=8086, debug=False)
