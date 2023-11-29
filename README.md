# VisDiff: Describing Differences in Image Sets with Natural Language

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![Pytorch](https://img.shields.io/badge/Pytorch-2.1-red.svg)](https://pytorch.org/get-started/previous-versions/#v21)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

This repo provides the PyTorch source code of our paper: [VisDiff: Describing Differences in Image Sets with Natural Language](.) (Under Review in CVPR 2024)

## 🔮 Abstract

How do two sets of images differ? Discerning set-level differences is crucial for understanding model behaviors and analyzing datasets, yet manually sifting through thousands of images is impractical. To aid in this discovery process, we explore the task of automatically describing the differences between two **sets** of images, which we term Set Difference Captioning. This task takes in image sets $\mathcal{D}_A$ and $\mathcal{D}_B$, and outputs a description that is more often true on $\mathcal{D}_A$ than $\mathcal{D}_B$. We outline a two-stage approach that first proposes candidate difference descriptions from image sets and then re-ranks the candidates by checking how well they can differentiate the two sets. We introduce VisDiff, which first captions the images and prompts a language model to propose candidate descriptions, then re-ranks these descriptions using CLIP. To evaluate VisDiff, we collect VisDiffBench, a dataset with 187 paired image sets with ground truth difference descriptions. We apply VisDiff to various domains, such as comparing datasets (e.g., ImageNet vs. ImageNetV2), comparing classification models (e.g., zero-shot CLIP vs. supervised ResNet), characterizing differences between generative models (e.g., StableDiffusionV1 and V2), and discovering what makes images memorable. Using VisDiff, we are able to find interesting and previously unknown differences in datasets and models, demonstrating its utility in revealing nuanced insights.

## 🚀 Getting Started

### 1. Environments

- Sign up or login to your [Weights and Biases Account](https://wandb.ai).

- Install required dependencies using Conda:
  ```bash
  conda env create -f environment.yml
  ```

- Activate the environment:
  ```bash
  conda activate visdiff
  ```

### 2. Datasets

We collect VisDiffBench ([Link](https://drive.google.com/file/d/1vghFd0rB5UTBaeR5rdxhJe3s7OOdRtkY)), which consists of 187 paired image sets with ground truth difference descriptions.

VisDiffBench is collected from the following datasets:

- PairedImageSets ([Collection Code](./data/pairedimagesets/)) 
- [ImageNetR](https://github.com/hendrycks/imagenet-r)
- [ImageNet*](https://huggingface.co/datasets/madrylab/imagenet-star)

### 3. Servers

We unify all the LLMs, VLMs, and CLIP to API servers for faster inference. Follow the instructions in [serve](./serve/README.md) to start these servers.

## 💼 Workflow

### 1. Convert Datasets

If you use VisDiffBench, you can skip this step. We provide all the converted CSVs in [data](./data/) folder.

If you want to use your own datasets, you can convert the dataset to CSV format with two required columns `path` and `group_name`. 

### 2. Define Configs

To describe the differences between two datasets, we need a [proposer](components/proposer.py), a [validator](components/validator.py), and an [evaluator](components/evaluator.py), which have different arguments.

We put all the base arguments in [base.yaml](configs/base.yaml) and specific arguments in [example.yaml](configs/example.yaml).

### 3. Describe Differences

We can run the following command to describe the differences between two datasets and log everything to Weights and Biases.

```bash
python main.py --config configs/example.yaml
```

## 💎 Applications

For each application, we provide the corresponding codes in [applications](applications/) folder.

## 🎯 Citation

If you use this repo in your research, please cite it as follows:
```
@inproceedings{
  anonymous2024visdiff,
  title={VisDiff: Describing Differences in Image Sets with Natural Language},
  author={Anonymous},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024},
  note={Under Review}
}
```