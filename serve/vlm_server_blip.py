import logging

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
    name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
)
logging.info("Model loaded successfully!")


@app.route("/", methods=["POST"])
def interact_with_blip():
    if "image" not in request.files:
        return jsonify({"error": "Image not provided"}), 400

    if "text" not in request.form:
        return jsonify({"error": "Text not provided"}), 400

    raw_image = Image.open(request.files["image"]).convert("RGB")
    with torch.no_grad():
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        result = model.generate({"image": image, "prompt": request.form["text"]})[0]

    return jsonify({"input": request.form["text"], "output": result})


if __name__ == "__main__":
    logging.info("Server is running!")
    app.run(host="0.0.0.0", port=8082, debug=False)
