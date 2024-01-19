import json
import logging
import threading
import base64

logging.basicConfig(level=logging.INFO)

import os
from typing import Dict, List

import lmdb
import requests
import openai

from serve.global_vars import BLIP_FEATURE_URL, BLIP_URL, LLAVA_URL, VLM_CACHE_FILE
from serve.utils_general import get_from_cache, save_to_cache

if not os.path.exists(VLM_CACHE_FILE):
    os.makedirs(VLM_CACHE_FILE)

vlm_cache = lmdb.open(VLM_CACHE_FILE, map_size=int(1e11))
openai.api_key = os.environ["OPENAI_API_KEY"]

def get_embed_caption_blip(
    sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
) -> List[str]:
    key = json.dumps([sampled_dataset1, sampled_dataset2, 1])
    cached_value = get_from_cache(key, vlm_cache)
    if cached_value is not None:
        logging.debug(f"VLM Cache Hit")
        cached_value = json.loads(cached_value)
        return cached_value

    try:
        response = requests.post(
            BLIP_FEATURE_URL,
            data={
                "dataset1": json.dumps(sampled_dataset1),
                "dataset2": json.dumps(sampled_dataset2),
            },
        ).json()
        output = response["output"]
        save_to_cache(key, json.dumps(output), vlm_cache)
        return output
    except Exception as e:
        logging.error(f"VLM Error: {e}")
    return ["VLM Error: Cannot get response."]

def get_image_base64(image_path):
    with open(image_path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')

def get_vlm_output(image: str, prompt: str, model: str) -> str:
    key = json.dumps([model, image, prompt])
    cached_value = get_from_cache(key, vlm_cache)
    if cached_value is not None:
        logging.debug(f"VLM Cache Hit")
        return cached_value

    if model in ["blip", "llava"]:
        files = {"image": open(image, "rb").read()}
        text_data = {"text": prompt}
        url = {
            "blip": BLIP_URL,
            "llava": LLAVA_URL,
        }[model]

        try:
            response = requests.post(url, data=text_data, files=files).json()
            output = response["output"]
            save_to_cache(key, output, vlm_cache)
            return output
        except Exception as e:
            logging.error(f"VLM Error: {e}")
            return "VLM Error: Cannot get response."

    elif model == "gpt-4-vision-preview":  # Add GPT-4V support
        base64_image = get_image_base64(image)
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 300
        }

        try:
            completion = openai.ChatCompletion.create(**payload)
            response = completion["choices"][0]["message"]["content"]
            save_to_cache(key, response, vlm_cache)
            return response
        except Exception as e:
            logging.error(f"VLM Error: {e}")
            return "VLM Error: Cannot get response."
    else:
        raise NotImplementedError(f"VLM model {model} not implemented.")


def captioning(image: str, model: str) -> str:
    caption = get_vlm_output(image, "Describe this image in detail.", model)
    return caption


def vqa(image: str, question: str, model: str) -> str:
    answer = get_vlm_output(image, question, model)
    return answer


def test_get_vlm_output():
    image = "data/teaser.png"
    model = "blip"

    caption = captioning(image, model)
    print(f"{caption=}")
    question = "Is there a table in the image?"
    answer = vqa(image, question, model)
    print(f"{answer=}")

    model = "gpt-4-vision-preview"

    caption = captioning(image, model)
    print(f"{caption=}")
    question = "Is there a table in the image?"
    answer = vqa(image, question, model)
    print(f"{answer=}")

    model = "llava"

    caption = captioning(image, model)
    print(f"{caption=}")
    question = "Is there a table in the image?"
    answer = vqa(image, question, model)
    print(f"{answer=}")


def test_get_vlm_output_parallel():
    threads = []

    for _ in range(3):
        thread = threading.Thread(target=test_get_vlm_output)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    test_get_vlm_output()
    # test_get_vlm_output_parallel()
