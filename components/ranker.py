import random
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, trange

import wandb
from serve.utils_clip import get_embeddings
from serve.utils_llm import get_llm_output
from serve.utils_vlm import get_vlm_output


def plot_distributions(similarity_A_C, similarity_B_C, hypothesis=""):
    """
    Plots the distributions of cos sim to hypothesis for each group.
    """
    # Convert arrays to 1D if they're 2D
    similarity_A_C = np.array(similarity_A_C).ravel()
    similarity_B_C = np.array(similarity_B_C).ravel()

    # Create a combined list of all scores and a list of labels to indicate group membership
    all_scores = list(similarity_A_C) + list(similarity_B_C)
    labels = ["Group A"] * len(similarity_A_C) + ["Group B"] * len(similarity_B_C)

    # Create a DataFrame for seaborn plotting
    df = pd.DataFrame({"Group": labels, "Similarity to C": all_scores})

    # Set up the figure with 3 subplots
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

    # Histogram
    ax[0].hist(similarity_A_C, bins=30, alpha=0.5, label="Group A", density=True)
    ax[0].hist(similarity_B_C, bins=30, alpha=0.5, label="Group B", density=True)
    ax[0].set_title(f"Histogram of Cosine Similarities to \n{hypothesis}")
    ax[0].set_ylabel("Density")
    ax[0].legend()

    # KDE plot
    sns.kdeplot(similarity_A_C, fill=True, ax=ax[1], label="Group A")
    sns.kdeplot(similarity_B_C, fill=True, ax=ax[1], label="Group B")
    ax[1].set_title(
        f"Kernel Density Estimation of Cosine Similarities to \n{hypothesis}"
    )
    ax[1].set_ylabel("Density")

    # Boxplot
    sns.boxplot(x="Group", y="Similarity to C", data=df, ax=ax[2])
    ax[2].set_title(f"Boxplot of Cosine Similarities to \n{hypothesis}")

    # Adjust layout
    plt.tight_layout()
    return fig


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
    return percent_correct_a - percent_correct_b


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


def t_test(d_A, d_B):
    d_A = np.array(d_A)
    d_B = np.array(d_B)

    # Assuming you've already defined your similarity scores d_A and d_B
    t_stat, p_value = ttest_ind(d_A, d_B, equal_var=False)

    # Decision
    alpha = 0.05
    if p_value < alpha:
        # print("** Reject the null hypothesis - there's a significant difference between the groups. **")
        return True, p_value
    else:
        # print("Fail to reject the null hypothesis - there's no significant difference between the groups.")
        return False, p_value


class Ranker:
    def __init__(self, args: Dict):
        self.args = args

    def score_hypothesis(self, hypothesis: str, dataset: List[dict]) -> List[float]:
        raise NotImplementedError

    def rerank_hypotheses(
        self, hypotheses: List[str], dataset1: List[dict], dataset2: List[dict]
    ) -> List[dict]:
        if len(dataset1) > self.args["max_num_samples"]:
            random.seed(self.args["seed"])
            dataset1 = random.sample(dataset1, self.args["max_num_samples"])
        if len(dataset2) > self.args["max_num_samples"]:
            random.seed(self.args["seed"])
            dataset2 = random.sample(dataset2, self.args["max_num_samples"])

        scored_hypotheses = []
        for hypothesis in tqdm(hypotheses):
            scores1 = self.score_hypothesis(hypothesis, dataset1)
            scores2 = self.score_hypothesis(hypothesis, dataset2)

            metrics = self.compute_metrics(scores1, scores2, hypothesis)
            scored_hypotheses.append(metrics)
        scored_hypotheses = sorted(
            scored_hypotheses, key=lambda x: x["auroc"], reverse=True
        )
        return scored_hypotheses

    def compute_metrics(
        self, scores1: List[float], scores2: List[float], hypothesis: str
    ) -> dict:
        metrics = {}
        metrics["hypothesis"] = hypothesis
        metrics["score1"] = np.mean(scores1)
        metrics["score2"] = np.mean(scores2)
        metrics["diff"] = metrics["score1"] - metrics["score2"]
        metrics["t_stat"], metrics["p_value"] = t_test(scores1, scores2)
        metrics["auroc"] = compute_auroc(scores1, scores2)
        metrics["correct_delta"] = classify(
            scores1, scores2, threshold=self.args["classify_threshold"]
        )
        metrics["distribution"] = wandb.Image(
            plot_distributions(scores1, scores2, hypothesis=hypothesis)
        )
        return metrics


class CLIPRanker(Ranker):
    def __init__(self, args: Dict):
        super().__init__(args)

    def score_hypothesis(self, hypothesis: str, dataset: List[dict]) -> List[float]:
        image_features = get_embeddings(
            [item["path"] for item in dataset], self.args["clip_model"], "image"
        )
        text_features = get_embeddings([hypothesis], self.args["clip_model"], "text")
        similarity = image_features @ text_features.T
        scores = similarity.squeeze(1).tolist()
        return scores


class VLMRanker(Ranker):
    def __init__(self, args: Dict):
        super().__init__(args)

    def score_hypothesis(self, hypothesis: str, dataset: List[dict]) -> List[float]:
        scores = []
        invalid_scores = []
        for i in trange(0, len(dataset)):
            item = dataset[i]
            prompt = f"Does this image contain {hypothesis.replace('and ', '')}?"  # TODO: why this prompt
            output = get_vlm_output(item["path"], prompt, self.args["model"])
            if "yes" in output.lower():
                scores.append(1)
            elif "no" in output.lower():
                scores.append(0)
            else:
                invalid_scores.append(output)
        print(f"Percent Invalid {len(invalid_scores) / len(dataset)}")
        return scores


class LLMRanker(Ranker):
    def __init__(self, args: Dict):
        super().__init__(args)

    def score_hypothesis(self, hypothesis: str, dataset: List[dict]) -> List[float]:
        scores = []
        invalid_scores = []
        for i in trange(0, len(dataset)):
            item = dataset[i]
            caption = (
                get_vlm_output(
                    item["path"],
                    self.args["captioner_prompt"],
                    self.args["captioner_model"],
                )
                .replace("\n", " ")
                .strip()
            )
            prompt = f"""Given a caption and a concept, respond with yes or no.
Here are 5 examples for the concept "spider and a flower":
INPUT: a spider sitting on top of a purple flower
OUTPUT: yes
INPUT: a yellow and black spider with a web in the background
OUTPUT: no
INPUT: a arachnid with a white flower
OUTPUT: yes
INPUT: a spider is walking on the ground in the grass
OUTPUT: no
INPUT: two yellow and black spiders
OUTPUT: no

Here are 6 examples for the concept "an ipod in the forest":
INPUT: a smartphone in the forest
OUTPUT: yes
INPUT: a white apple ipad sitting on top of a wooden table
OUTPUT: no
INPUT: an ipod near some trees
OUTPUT: yes
INPUT: a smartphone with apps
OUTPUT: no
INPUT: a pink mp3 player sitting on top of a book
OUTPUT: no
INPUT: an ipod sitting on a white surface
OUTPUT: no

Given the caption "{caption}" and the concept "{hypothesis}", respond with either the word yes or no ONLY.
OUTPUT:"""
            output = get_llm_output(prompt, self.args["model"])
            if "yes" in output.lower():
                scores.append(1)
            elif "no" in output.lower():
                scores.append(0)
            else:
                invalid_scores.append(output)
        print(f"Percent Invalid {len(invalid_scores) / len(dataset)}")
        return scores


class NullRanker(Ranker):
    def __init__(self, args: Dict):
        super().__init__(args)

    def score_hypothesis(self, hypothesis: str, dataset: List[dict]) -> List[float]:
        return [0.0] * len(dataset)


def test_rankers():
    args = {
        "clip_model": "ViT-bigG-14",
        "clip_dataset": "laion2b_s39b_b160k",
        "model": "llava",
        "batch_size": 32,
        "classify_threshold": 0.3,
    }

    dataset = pd.read_csv("data/diffusion_plates.csv")
    dataset = dataset.to_dict("records")
    dataset1 = [item for item in dataset if item["set"] == "a_plate"][:20]
    dataset2 = [item for item in dataset if item["set"] == "a_dinner_plate"][:20]
    for item in dataset1 + dataset2:
        item["caption"] = get_vlm_output(item["path"], "Describe this image", "llava")

    hypotheses = ["A cat", "Food"]

    ranker_clip = CLIPRanker(args)
    scores = ranker_clip.rerank_hypotheses(hypotheses, dataset1, dataset2)
    print(scores)

    ranker_vlm = VLMRanker(args)
    scores = ranker_vlm.rerank_hypotheses(hypotheses, dataset1, dataset2)
    print(scores)

    ranker_llm = LLMRanker(args)
    scores = ranker_llm.rerank_hypotheses(hypotheses, dataset1, dataset2)
    print(scores)


if __name__ == "__main__":
    test_rankers()
