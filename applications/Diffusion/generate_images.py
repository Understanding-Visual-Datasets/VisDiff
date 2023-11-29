# from torch import autocast
import argparse
import os

import pandas as pd
import torch
import wandb
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline


def main(args):
    # if args.wandb_silent:
    #     os.environ['WANDB_SILENT']="true"

    device = "cuda"

    print(f"Prompts: {args.prompts}")

    # assert args.num_images < 50, "num_images must be less than 50"
    latents = torch.load("data/diffusion/latents.pt", map_location=device)

    wandb.init(project="Text-2-Image-ImDiff", config=vars(args))

    with open("./data/negative_prompts.txt", "r") as f:
        negative_prompts = [line.replace("\n", "") for line in f.readlines()]
    negative_prompt = ", ".join(negative_prompts)
    print(f"Negative Prompt: {negative_prompt}")

    for model_id in args.model_id:
        # Use the Euler scheduler here instead
        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            requires_safety_checker=False,
            safety_checker=None,
        )
        pipe = pipe.to("cuda")
        if args.prompts == ["PartiPrompts"]:
            parti_prompts = pd.read_csv("./data/parti-prompts.csv")
            prompts = parti_prompts["Prompt"].tolist()
        elif args.prompts == ["DiffusionDB"]:
            with open("./data/diffusion/diffusiondb_big", "r") as f:
                prompts = [line.replace("\n", "") for line in f.readlines()]
        else:
            prompts = args.prompts
        for p, prompt in enumerate(prompts):
            if os.path.exists(
                f'{args.save_dir}/{prompt.replace(" ", "_").replace(".", "")}/{model_id}'
            ):
                continue
            print(f"Generating images for prompt: {prompt}")
            with torch.autocast("cuda"):
                # split up into batches of 5
                images = []
                step = min([5, args.n])
                for i in range(0, args.n, step):
                    images += pipe(
                        prompt,
                        negative_prompt=negative_prompt,
                        num_images_per_prompt=step,
                        guidance_scale=7.5,
                        latents=latents[i : i + step],
                    ).images

            for s, i in enumerate(images):
                save_dir = f'{args.save_dir}/{prompt.replace(" ", "_").replace(".", "")[:100]}/{model_id}'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                i.save(f"{save_dir}/{s}.png")

            wandb.log({f"{model_id}-prompt": [wandb.Image(i) for i in images[:min([20, args.n])]]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Understanding")
    parser.add_argument("--prompts", type=str, nargs="+", help="prompts")
    parser.add_argument("--dataset", type=str, default="Cub2011", help="dataset")
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="save directory",
    )
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument(
        "--n", type=int, default=50, help="number of images to generate"
    )
    parser.add_argument("--wandb-silent", action="store_true")
    parser.add_argument(
        "--model-id",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        nargs="+",
        help="huggingface model id",
    )
    args = parser.parse_args()
    main(args)
