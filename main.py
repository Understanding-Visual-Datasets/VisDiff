import logging
from typing import Dict, List, Tuple

import click
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
from components.evaluator import GPTEvaluator, NullEvaluator
from components.proposer import (
    LLMProposer,
    LLMProposerDiffusion,
    VLMFeatureProposer,
    VLMProposer,
)
from components.ranker import CLIPRanker, LLMRanker, NullRanker, VLMRanker
from components.sampling import classifier_sampler


def load_config(config: str) -> Dict:
    base_cfg = OmegaConf.load("configs/base.yaml")
    cfg = OmegaConf.load(config)
    final_cfg = OmegaConf.merge(base_cfg, cfg)
    args = OmegaConf.to_container(final_cfg)
    args["config"] = config
    if args["wandb"]:
        wandb.init(
            project=args["project"],
            name=args["data"]["name"],
            group=f'{args["data"]["group1"]} - {args["data"]["group2"]} ({args["data"]["purity"]})',
            config=args,
        )
    return args


def load_data(args: Dict) -> Tuple[List[Dict], List[Dict], List[str]]:
    data_args = args["data"]

    df = pd.read_csv(f"{data_args['root']}/{data_args['name']}.csv")

    if data_args["subset"]:
        old_len = len(df)
        df = df[df["subset"] == data_args["subset"]]
        print(
            f"Taking {data_args['subset']} subset (dataset size reduced from {old_len} to {len(df)})"
        )

    dataset1 = df[df["group_name"] == data_args["group1"]].to_dict("records")
    dataset2 = df[df["group_name"] == data_args["group2"]].to_dict("records")
    group_names = [data_args["group1"], data_args["group2"]]

    if data_args["purity"] < 1:
        logging.warning(f"Purity is set to {data_args['purity']}. Swapping groups.")
        assert len(dataset1) == len(dataset2), "Groups must be of equal size"
        n_swap = int((1 - data_args["purity"]) * len(dataset1))
        dataset1 = dataset1[n_swap:] + dataset2[:n_swap]
        dataset2 = dataset2[n_swap:] + dataset1[:n_swap]
    print(dataset1)
    return dataset1, dataset2, group_names


def propose(args: Dict, dataset1: List[Dict], dataset2: List[Dict]) -> List[str]:
    proposer_args = args["proposer"]
    proposer_args["seed"] = args["seed"]
    proposer_args["captioner"] = args["captioner"]

    proposer = eval(proposer_args["method"])(proposer_args)
    hypotheses, logs, images = proposer.propose(dataset1, dataset2)
    if args["wandb"]:
        wandb.log({"logs": wandb.Table(dataframe=pd.DataFrame(logs))})
        for i in range(len(images)):
            wandb.log(
                {
                    f"group 1 images ({dataset1[0]['group_name']})": images[i][
                        "images_group_1"
                    ],
                    f"group 2 images ({dataset2[0]['group_name']})": images[i][
                        "images_group_2"
                    ],
                }
            )
    return hypotheses


def rank(
    args: Dict,
    hypotheses: List[str],
    dataset1: List[Dict],
    dataset2: List[Dict],
    group_names: List[str],
) -> List[str]:
    ranker_args = args["ranker"]
    ranker_args["seed"] = args["seed"]

    ranker = eval(ranker_args["method"])(ranker_args)

    scored_hypotheses = ranker.rerank_hypotheses(hypotheses, dataset1, dataset2)
    if args["wandb"]:
        table_hypotheses = wandb.Table(dataframe=pd.DataFrame(scored_hypotheses))
        wandb.log({"scored hypotheses": table_hypotheses})
        for i in range(5):
            wandb.summary[f"top_{i + 1}_difference"] = scored_hypotheses[i][
                "hypothesis"
            ].replace('"', "")
            wandb.summary[f"top_{i + 1}_score"] = scored_hypotheses[i]["auroc"]

    scored_groundtruth = ranker.rerank_hypotheses(
        group_names,
        dataset1,
        dataset2,
    )
    if args["wandb"]:
        table_groundtruth = wandb.Table(dataframe=pd.DataFrame(scored_groundtruth))
        wandb.log({"scored groundtruth": table_groundtruth})

    return [hypothesis["hypothesis"] for hypothesis in scored_hypotheses]


def evaluate(args: Dict, ranked_hypotheses: List[str], group_names: List[str]) -> Dict:
    evaluator_args = args["evaluator"]

    evaluator = eval(evaluator_args["method"])(evaluator_args)

    metrics, evaluated_hypotheses = evaluator.evaluate(
        ranked_hypotheses,
        group_names[0],
        group_names[1],
    )

    if args["wandb"] and evaluator_args["method"] != "NullEvaluator":
        table_evaluated_hypotheses = wandb.Table(
            dataframe=pd.DataFrame(evaluated_hypotheses)
        )
        wandb.log({"evaluated hypotheses": table_evaluated_hypotheses})
        wandb.log(metrics)
    return metrics


@click.command()
@click.option("--config", help="config file")
def main(config):
    logging.info("Loading config...")
    args = load_config(config)
    # print(args)

    logging.info("Loading data...")
    dataset1, dataset2, group_names = load_data(args)
    # print(dataset1, dataset2, group_names)

    logging.info("test sampling...")
    if args["proposer"]["sampling_method"] == "classifier":
        model = classifier_sampler(
            dataset1, dataset2, model=args["proposer"]["classifier_model"]
        )
        exit(0)

    logging.info("Proposing hypotheses...")
    hypotheses = propose(args, dataset1, dataset2)
    # print(hypotheses)

    logging.info("Ranking hypotheses...")
    ranked_hypotheses = rank(args, hypotheses, dataset1, dataset2, group_names)
    # print(ranked_hypotheses)

    logging.info("Evaluating hypotheses...")
    metrics = evaluate(args, ranked_hypotheses, group_names)
    # print(metrics)


if __name__ == "__main__":
    main()
