import argparse
import logging
import os

import pandas as pd
import torch
from tqdm import tqdm
import open_clip
import torch.nn.functional as F
import wandb
from PIL import Image
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)

import helpers.method as method
from serve.utils_vlm import get_vlm_output
from utils import *
from components.validator import get_embedding_from_cache
        
parser = argparse.ArgumentParser(description="VisDiff")
parser.add_argument("--hypothesis", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--group", default='set')
parser.add_argument("--group_idxs", nargs='+')
parser.add_argument("--class_col", default='target')
parser.add_argument("--class_idxs", nargs='+')
parser.add_argument("--threshold", default=0.3, type=float, help='clip threshold')
parser.add_argument("--visualize", action='store_true', help='visualize the differences')
parser.add_argument("--visualize-num", type=int, default=10, help='visualize the differences')
parser.add_argument("--clip-model", type=str, default="ViT-bigG-14", help='save the results')
parser.add_argument("--clip-dataset", type=str, default="laion2b_s39b_b160k", help='save the results')
args = parser.parse_args()

df = pd.read_csv(f'./data/{args.dataset}.csv')
print(f"loaded dataset {args.dataset} of size {len(df)}")
if args.group_idxs:
    group_names = list(args.group_idxs)
else:
    group_names = df[args.group].unique().tolist()
run_name = "-".join(group_names)

wandb.init(project="VisDiff-EvalSingleHyp", name=args.hypothesis, group=args.dataset, config=args)
logging.info(f"Comparing set 1 = {group_names[0]} and set 2 = {group_names[1]}")

# if we want to pick certain groups
if args.group_idxs:
    old_len = len(df)
    df = df[df[args.group].isin(group_names)].reset_index(
        drop=True
    )
    # Convert 'group' column to a categorical type with specific ordering
    df[args.group] = pd.Categorical(
        df[args.group],
        categories=group_names,
        ordered=True,
    )

    # Sort DataFrame based on custom order
    df = df.sort_values(by=args.group)
    # trainset = Subset(trainset, captions_df.index.tolist())
    logging.info(
        f"Filtered captions from {old_len} to {len(df)} in groups {group_names}"
    )

if args.class_idxs:
    old_len = len(df)
    df = df[df[args.class_col].isin(args.class_idxs)].reset_index(
        drop=True
    )
    print(
        f"keeping class idxs {args.class_idxs} before {old_len} after {df[args.group].value_counts()}"
    )

        
model, _, preprocess = open_clip.create_model_and_transforms(
args.clip_model, pretrained=args.clip_dataset)
# model, _, preprocess = open_clip.create_model_and_transforms(
# 'ViT-L-14', pretrained='openai')
#  ('ViT-L-14', 'openai'),
smodel = model.to('cuda').eval()
tokenizer = open_clip.get_tokenizer(args.clip_model)
# tokenizer = open_clip.get_tokenizer('ViT-L-14')

def get_embeddings(dataset):
    """
    Returns the embeddings for the dataset.
    """
    embeddings = []
    for row in tqdm(dataset, total=len(dataset), desc="Computing CLIP embeddings"):
        embedding = get_embedding_from_cache(row['path'], model, preprocess, args.clip_model, args.clip_dataset, 'cuda')
        embeddings.append(np.squeeze(embedding))
    print(f"Computed clip embeddngs {np.array(embeddings).shape}")
    return torch.Tensor(np.array(embeddings)).to('cuda')

def classify(similarity_A_C, similarity_B_C, threshold=0.3):
    """
    Given two arrays of cos sim scores, classify each item of each group as containing concept C or not.
    Return P(hyp in A) - P(hyp in B)
    """
    similarity_A_C = np.array(similarity_A_C)
    similarity_B_C = np.array(similarity_B_C)
    # print(
    #     f"avg(cos sim A, cos sim B) = {[np.mean(similarity_A_C), np.mean(similarity_B_C)]} \t Max(cos sim A, cos sim B) = {[np.max(similarity_A_C), np.max(similarity_B_C)]}"
    # )
    percent_correct_a = sum(similarity_A_C > threshold) / len(similarity_A_C)
    percent_correct_b = sum(similarity_B_C > threshold) / len(similarity_B_C)
    # print(f"Percent correct A, B {[percent_correct_a, percent_correct_b]}")
    return percent_correct_a, percent_correct_b

def compute_auroc(similarity_A_C, similarity_B_C):
    similarity_A_C = np.array(similarity_A_C)
    similarity_B_C = np.array(similarity_B_C)

    # Create labels based on the sizes of the input arrays
    labels_A = [1] * similarity_A_C.shape[0]
    labels_B = [0] * similarity_B_C.shape[0]

    # Concatenate scores and labels using numpy's concatenate
    all_scores = np.concatenate([similarity_A_C, similarity_B_C], axis=0).ravel()
    all_labels = labels_A + labels_B

    # Compute AUROC
    auroc = roc_auc_score(all_labels, all_scores)
    return auroc

image_features_a, image_features_b = get_embeddings(df[df[args.group] == group_names[0]].to_dict('records')), get_embeddings(df[df[args.group] == group_names[1]].to_dict('records'))
image_features = get_embeddings(df.to_dict('records'))
text_features = model.encode_text(tokenizer([args.hypothesis]).to('cuda'))
text_features = F.normalize(text_features, dim=-1)
similarity = image_features @ text_features.T
scores = similarity.squeeze(1).detach().cpu().numpy().tolist()
df['score'] = scores

similarity_A_C = image_features_a @ text_features.T
similarity_A_C = similarity_A_C.squeeze(1).detach().cpu().numpy().tolist()
similarity_B_C = image_features_b @ text_features.T
similarity_B_C = similarity_B_C.squeeze(1).detach().cpu().numpy().tolist()
percent_correct_a, percent_correct_b = classify(similarity_A_C, similarity_B_C, args.threshold)
auroc = compute_auroc(similarity_A_C, similarity_B_C)
print(f"Percent correct A, B {[percent_correct_a, percent_correct_b]}")
print(f"AUROC {auroc}")

df['contains_hypothesis'] = df['score'] > args.threshold
group_idxs = '-'.join(args.group_idxs).replace(' ', '_')
class_idxs = '-'.join(args.class_idxs).replace(' ', '_') if args.class_idxs else 'all'
if not os.path.exists(f'./results/{args.dataset}/{group_idxs}/{class_idxs}'):
    os.makedirs(f'./results/{args.dataset}/{group_idxs}/{class_idxs}')
df.to_csv(f'./results/{args.dataset}/{group_idxs}/{class_idxs}/{args.hypothesis.replace(" ", "_")}.csv')
# log as table
wandb.log({"per-image-scores": wandb.Table(dataframe=df)})
wandb.summary['num group A'] = len(df[(df[args.group] == group_names[0]) & (df['contains_hypothesis'] == True)])
wandb.summary['num group B'] = len(df[(df[args.group] == group_names[1]) & (df['contains_hypothesis'] == True)])
wandb.summary['percent group A'] = wandb.summary['num group A'] / len(df[df[args.group] == group_names[0]])
wandb.summary['percent group B'] = wandb.summary['num group B'] / len(df[df[args.group] == group_names[1]])
wandb.summary['percent diff'] = wandb.summary['percent group A'] - wandb.summary['percent group B']
wandb.summary['auroc'] = auroc
if args.visualize:
    # visualize the top images from each group by score
    df['score'] = df['score'].astype(float)
    df = df.sort_values(by=['score'], ascending=False)
    for g in df[args.group].unique():
        group_df = df[df[args.group] == g].head(args.visualize_num)
        wandb.log(
            {
                f"image_caption_pairs_group-{g}": [
                    wandb.Image(
                        Image.open(list(group_df["path"])[i]).resize((224, 224))
                    )
                    for i in range(len(group_df))
                ]
            }
        )