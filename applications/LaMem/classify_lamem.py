import clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)


def classify_images(dataframe, group_a_prompts, group_b_prompts):
    # Combine all prompts into one list and preprocess them
    all_prompts = group_a_prompts + group_b_prompts
    text_tokens = clip.tokenize(all_prompts).to(device)

    # Process each image in the dataframe
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        # Load and preprocess the image
        image_path = row["path"]
        image = (
            preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        )

        # Calculate the similarity with each prompt
        with torch.no_grad():
            # Cosine similarity as logits
            logits_per_image, _ = model(image, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # Get the most similar prompt and its group
        max_index = probs.argmax()
        prompt_prediction = all_prompts[max_index]
        group_prediction = "A" if max_index < len(group_a_prompts) else "B"

        # Update the dataframe
        dataframe.at[index, "prompt_prediction"] = prompt_prediction
        dataframe.at[index, "group_prediction"] = group_prediction

    return dataframe


# Example usage
df = pd.read_csv("data/lamem_25_75.csv")
group_a_prompts = [
    "close-up of individual people",
    "use of accessories or personal items",
    "tattoos on human skin",
    "close-up on individuals",
    "humorous or funny elements",
    "artistic or unnaturally altered human features",
    "humorous elements",
    "detailed description of tattoos",
    "fashion and personal grooming activities",
    "pop culture references",
    "collectibles or hobbies",
    "light-hearted or humorous elements",
    "themed costumes or quirky outfits",
    "animated or cartoonish characters",
    "emphasis on fashion or personal style",
    "close-up of objects or body parts",
    "close-up facial expressions",
    "unconventional use of everyday items",
    "images with a playful or humorous element",
    "focus on specific body parts",
    "silly or humorous elements",
    "people in casual or humorous situations",
    "detailed description of attire",
    "quirky and amusing objects",
    "humorous or playful expressions",
]
group_b_prompts = [
    "Sunsets and sunrises",
    "serene beach settings",
    "sunset or nighttime scenes",
    "agricultural fields",
    "clear daytime outdoor settings",
    "landscapes with water bodies",
    "images captured during different times of day and night",
    "Beautiful skies or sunsets",
    "abandoned or isolated structures",
    "natural elements like trees and water",
    "urban cityscapes",
    "various weather conditions",
    "Afar shots of buildings or architectural structures",
    "outdoor landscapes",
    "cityscapes",
    "Cityscapes and urban environments",
    "Scenic outdoor landscapes",
    "landscapes with mountains",
    "Picturesque mountain views",
    "expansive outdoor landscapes",
    "Scenic landscapes or nature settings",
    "Serene and tranquil environments",
    "scenic landscapes",
    "scenes with a serene and peaceful atmosphere",
]

classified_df = classify_images(df, group_a_prompts, group_b_prompts)
classified_df.to_csv("results/lamem_25_75_classified.csv", index=False)
