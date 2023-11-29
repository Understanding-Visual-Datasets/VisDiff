import io
import json
import os
from abc import ABC, abstractmethod

import clip
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# Fetch ImageNet class names
IMAGENET_CLASSES_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
if not os.path.exists("imagenet_class_index.json"):
    r = requests.get(IMAGENET_CLASSES_URL)
    with open("imagenet_class_index.json", "wb") as f:
        f.write(r.content)
class_idx = json.load(open("imagenet_class_index.json", "r"))
IDX_TO_LABEL = [class_idx[str(k)][1].replace("_", " ") for k in range(len(class_idx))]


class BaseModel(ABC):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def get_prediction(self, image: str) -> str:
        pass


class CLIPModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model, self.transform = self._load_model()
        self.model.eval()
        text_inputs = torch.cat(
            [clip.tokenize(f"a photo of a {c}") for c in IDX_TO_LABEL]
        )
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_inputs.to(self.device))
            self.text_features = F.normalize(self.text_features, p=2, dim=1)

    def _load_model(self):
        return clip.load("ViT-B/32")

    def get_prediction(self, image: str) -> str:
        pil_image = Image.open(image).convert("RGB")
        tensor_image = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Encode the list of ImageNet class names

        with torch.no_grad():
            image_features = self.model.encode_image(tensor_image)
            image_features = F.normalize(image_features, p=2, dim=1)

            # Find the top 1 most similar label for the image
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            _, indices = similarity[0].topk(1)

            return IDX_TO_LABEL[indices[0].item()]


class ResNet50Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = self._load_model()
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_model(self):
        model = models.resnet50(pretrained=True)
        model.eval().to(self.device)
        return model

    def get_prediction(self, image: str) -> str:
        pil_image = Image.open(image).convert("RGB")
        tensor_image = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor_image)
            _, predicted = outputs.max(1)

            return IDX_TO_LABEL[predicted.item()]


class ModelFactory:
    @staticmethod
    def get_model(model_name: str = "clip_vitb32_zeroshot") -> BaseModel:
        assert model_name in [
            "clip_vitb32_zeroshot",
            "resnet50_supervised",
        ], f"Model does not support {model_name}"

        if model_name == "clip_vitb32_zeroshot":
            return CLIPModel()
        else:
            return ResNet50Model()


if __name__ == "__main__":
    image = "../../../_deprecated/initial_attempt/532_v1/ILSVRC2012_val_00000241.JPEG"

    model = ModelFactory.get_model("clip_vitb32_zeroshot")
    classname = model.get_prediction(image)
    print(classname)

    model = ModelFactory.get_model("resnet50_supervised")
    classname = model.get_prediction(image)
    print(classname)
