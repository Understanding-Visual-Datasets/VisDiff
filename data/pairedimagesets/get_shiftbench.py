import json
import logging
import os

from PIL import Image
from tqdm import tqdm


def crawl():
    from icrawler.builtin import BingImageCrawler, GoogleImageCrawler

    os.remove("crawler.log")
    logging.basicConfig(filename="crawler.log", level=logging.INFO)

    for filename in ["easy.jsonl", "medium.jsonl", "hard.jsonl"]:
        data = [json.loads(line) for line in open(f"webcrawl/{filename}")]
        for idx, item in tqdm(enumerate(data)):
            set1, set2 = item["set1"], item["set2"]

            logging.info(f"##### Processing {filename} {idx}_1 ({set1}) #####")
            google_crawler = BingImageCrawler(
                feeder_threads=2,
                parser_threads=4,
                downloader_threads=16,
                storage={
                    "root_dir": f"webcrawl/{filename.replace('.jsonl', '')}/{idx}_1"
                },
            )
            google_crawler.crawl(keyword=set1, max_num=200)

            logging.info(f"##### Processing {filename} {idx}_2 ({set2}) #####")
            google_crawler = BingImageCrawler(
                feeder_threads=2,
                parser_threads=4,
                downloader_threads=16,
                storage={
                    "root_dir": f"webcrawl/{filename.replace('.jsonl', '')}/{idx}_2"
                },
            )
            google_crawler.crawl(keyword=set2, max_num=200)

    # google_crawler = BingImageCrawler(
    #     feeder_threads=1, parser_threads=2, downloader_threads=4, storage={"root_dir": "7"}
    # )
    # google_crawler.crawl(keyword='Sunset over Santorini, Greece', filters={'license': 'creativecommons'}, max_num=20)


def process_image_to_jpg(input_path, output_path, resolution=512):
    """
    Convert and resize the image to JPG format with max dimension of 512.

    Parameters:
    - input_path (str): Path to the source image.
    - output_path (str): Path to save the processed JPG image.

    Returns:
    None
    """

    # Open the image
    with Image.open(input_path) as img:
        img = img.convert("RGB")

        # Get the aspect ratio
        aspect_ratio = img.width / img.height

        # Determine new dimensions based on aspect ratio
        if img.width > img.height:
            new_width = resolution
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = resolution
            new_width = int(new_height * aspect_ratio)

        # Resize the image
        img_resized = img.resize((new_width, new_height), resample=Image.LANCZOS)

        # Save as JPG
        img_resized.save(output_path, "JPEG")


def release(n_sample=100):
    # make a dir named "VisDiffBench"
    os.makedirs("VisDiffBench", exist_ok=True)
    logs = open("webcrawl/crawler_bing_200.log").readlines()

    all_logs = []
    current_logs = []
    current_idx = 1
    for log in logs:
        if log.startswith("INFO:root:##### Processing"):
            all_logs.append(current_logs)
            current_logs = []
            current_idx = 1
        else:
            if log.startswith("INFO:downloader:image #"):
                parsed_log = (
                    log.strip().replace("INFO:downloader:image #", "").split("\t")
                )
                assert len(parsed_log) == 2
                assert int(parsed_log[0]) == current_idx
                current_idx += 1
                current_logs.append(parsed_log[1])
    all_logs.append(current_logs)  # add the last one
    current_logs = []
    all_logs = all_logs[1:]  # remove the first empty array
    # print(len(all_logs), len(all_logs[0]), all_logs[0][:10])

    for diff_idx, difficulty in enumerate(["easy", "medium", "hard"]):
        data = [json.loads(line) for line in open(f"webcrawl/{difficulty}.jsonl")]
        for idx, item in tqdm(enumerate(data)):
            os.makedirs(f"VisDiffBench/{difficulty}/{idx}_1", exist_ok=True)
            os.makedirs(f"VisDiffBench/{difficulty}/{idx}_2", exist_ok=True)

            set1_images = sorted(os.listdir(f"webcrawl/{difficulty}/{idx}_1"))
            set2_images = sorted(os.listdir(f"webcrawl/{difficulty}/{idx}_2"))
            if not (len(set1_images) >= n_sample and len(set2_images) >= n_sample):
                print(f"{difficulty}/{idx} has less than {n_sample} images")
                print(f"set1: {len(set1_images)}, set2: {len(set2_images)}")

            set_1_images_sampled = set1_images[:n_sample]
            set_2_images_sampled = set2_images[:n_sample]
            # print(set_1_images_sampled[:10])
            # input()

            # copy these files to new folder
            for image_idx, image in enumerate(set_1_images_sampled):
                assert image_idx + 1 == int(image.split(".")[0])
                process_image_to_jpg(
                    f"webcrawl/{difficulty}/{idx}_1/{image}",
                    f"VisDiffBench/{difficulty}/{idx}_1/{image.split('.')[0]}.jpg",
                )

            for image_idx, image in enumerate(set_2_images_sampled):
                assert image_idx + 1 == int(image.split(".")[0])
                process_image_to_jpg(
                    f"webcrawl/{difficulty}/{idx}_2/{image}",
                    f"VisDiffBench/{difficulty}/{idx}_2/{image.split('.')[0]}.jpg",
                )

            item["set1_images"] = [
                f"{difficulty}/{idx}_1/{i + 1:06d}.jpg" for i in range(n_sample)
            ]
            item["set2_images"] = [
                f"{difficulty}/{idx}_2/{i + 1:06d}.jpg" for i in range(n_sample)
            ]
            item["set1_images_url"] = all_logs[diff_idx * n_sample + idx * 2][:n_sample]
            item["set2_images_url"] = all_logs[diff_idx * n_sample + idx * 2 + 1][
                :n_sample
            ]

        # write jsonl
        with open(f"VisDiffBench/{difficulty}.jsonl", "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    # crawl()
    release()
