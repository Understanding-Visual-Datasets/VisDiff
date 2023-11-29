import json
import logging
from typing import List

import lmdb
import numpy as np
import requests

from serve.global_vars import CLIP_CACHE_FILE, CLIP_URL
from serve.utils_general import get_from_cache, save_to_cache

clip_cache = lmdb.open(CLIP_CACHE_FILE, map_size=int(1e11))


def get_embeddings(inputs: List[str], model: str, modality: str) -> np.ndarray:
    input_to_embeddings = {}
    for inp in inputs:
        key = json.dumps([inp, model])
        cached_value = get_from_cache(key, clip_cache)
        if cached_value is not None:
            logging.debug(f"CLIP Cache Hit")
            input_to_embeddings[inp] = json.loads(cached_value)

    uncached_inputs = [inp for inp in inputs if inp not in input_to_embeddings]

    if len(uncached_inputs) > 0:
        try:
            response = requests.post(
                CLIP_URL, data={modality: json.dumps(uncached_inputs)}
            ).json()
            print(type(response["embeddings"]), len(response["embeddings"]))
            for inp, embedding in zip(uncached_inputs, response["embeddings"]):
                input_to_embeddings[inp] = embedding
                key = json.dumps([inp, model])
                save_to_cache(key, json.dumps(embedding), clip_cache)
        except Exception as e:
            logging.error(f"CLIP Error: {e}")
            for inp in uncached_inputs:
                input_to_embeddings[inp] = None

    input_embeddings = [input_to_embeddings[inp] for inp in inputs]
    return np.array(input_embeddings)


if __name__ == "__main__":
    embeddings = get_embeddings(
        ["../_deprecated/initial_attempt/532_v1/ILSVRC2012_val_00000241.JPEG"],
        "ViT-bigG-14",
        "image",
    )
    print(embeddings)

    embeddings = get_embeddings(["shit", "haha", "hello world"], "ViT-bigG-14", "text")
    print(embeddings)
