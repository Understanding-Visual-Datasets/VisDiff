import hashlib
from typing import Dict, List, Optional

import lmdb
from PIL import Image


def resize_image(image: Image.Image, size=(256, 256)) -> Image.Image:
    return image.resize(size)


def merge_images_horizontally(images: List[Image.Image], gap: int = 10) -> Image.Image:
    imgs = [resize_image(image) for image in images]
    total_width = sum(img.width for img in imgs) + gap * (len(imgs) - 1)
    height = imgs[0].height

    merged = Image.new("RGB", (total_width, height))

    x_offset = 0
    for img in imgs:
        merged.paste(img, (x_offset, 0))
        x_offset += img.width + gap

    return merged


def merge_images_vertically(images: List[Image.Image], gap: int = 10) -> Image.Image:
    imgs = images
    total_height = sum(img.height for img in imgs) + gap * (len(imgs) - 1)
    width = max(img.width for img in imgs)

    merged = Image.new("RGB", (width, total_height))

    y_offset = 0
    for img in imgs:
        merged.paste(img, (0, y_offset))
        y_offset += img.height + gap

    return merged


def save_data_diff_image(dataset1: List[Dict], dataset2: List[Dict], save_path: str):
    assert len(dataset1) == len(dataset2), "Datasets must be of the same length"
    n_images = len(dataset1)

    # Load images into memory as PIL Image objects
    images_dataset1 = [Image.open(item["path"]) for item in dataset1]
    images_dataset2 = [Image.open(item["path"]) for item in dataset2]

    # Merge images from the same dataset horizontally
    merged_images_dataset1_first = merge_images_horizontally(
        images_dataset1[: n_images // 2]
    )
    merged_images_dataset1_second = merge_images_horizontally(
        images_dataset1[n_images // 2 :]
    )
    merged_images_dataset2_first = merge_images_horizontally(
        images_dataset2[: n_images // 2]
    )
    merged_images_dataset2_second = merge_images_horizontally(
        images_dataset2[n_images // 2 :]
    )

    # Merge the resulting images from different datasets vertically
    final_merged_image = merge_images_vertically(
        [
            merged_images_dataset1_first,
            merged_images_dataset1_second,
            merged_images_dataset2_first,
            merged_images_dataset2_second,
        ]
    )

    # Save the merged image
    final_merged_image.save(save_path)


def hash_key(key) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def get_from_cache(key: str, env: lmdb.Environment) -> Optional[str]:
    with env.begin(write=False) as txn:
        hashed_key = hash_key(key)
        value = txn.get(hashed_key.encode())
    if value:
        return value.decode()
    return None


def save_to_cache(key: str, value: str, env: lmdb.Environment):
    with env.begin(write=True) as txn:
        hashed_key = hash_key(key)
        txn.put(hashed_key.encode(), value.encode())
