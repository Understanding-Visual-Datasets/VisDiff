import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import wandb
import os
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import clip
import collections
import torchvision.transforms as transforms
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)

from utils import *
import models
import torch

parser = argparse.ArgumentParser(description='Dataset Understanding')
parser.add_argument('--config', default='configs/base.yaml', help="config file")
parser.add_argument('--run_name', required=True, help='name of wandb run')
parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                "(use dots for.nested=overrides)")
flags, unknown = parser.parse_known_args()

overrides = OmegaConf.from_cli(flags.overrides)
cfg       = OmegaConf.load(flags.config)
base      = OmegaConf.load('configs/base.yaml')
dataset_base = OmegaConf.load(cfg.base_config)
args      = OmegaConf.merge(base, dataset_base, cfg, overrides)
if len(unknown) > 0:
    print(unknown)
    config = nest_dict(read_unknowns(unknown))
    to_merge = OmegaConf.create(config)
    args = OmegaConf.merge(args, to_merge)
args.yaml = flags.config
args.run_name = flags.run_name

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.wandb_silent:
    os.environ['WANDB_SILENT']="true"

torch.manual_seed(args.seed)
np.random.seed(args.seed)

import torchvision.transforms as transforms
from PIL import Image
from utils import get_counts

class CSVDataset:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    def __init__(self, df, group_name, target_name, split='train'):
        self.df = df
        self.df['target_idx'] = self.df['target_idx'].astype(int)
        self.df['group_idx'] = self.df['group_idx'].astype(int)
        self.classes = self.class_names = list(self.df.drop_duplicates('target_idx').set_index('target_idx').sort_index()[target_name].to_dict().values())
        self.group_names = list(self.df.drop_duplicates('group_idx').set_index('group_idx').sort_index()[group_name].to_dict().values())
        self.df = df[df['split'] == split]
        self.samples = list(zip(self.df['path'], self.df['target_idx']))
        self.groups = list(self.df['group_idx'])
        self.transform = self.train_transform if split == 'train' else self.transform
        print(self.samples[:5])
        # self.class_weights = get_counts([s[1] for s in self.samples])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, target = self.samples[idx]
        img = Image.open(filename).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        group = self.groups[idx]
        return img, target, group, filename

ckpt_name = f'checkpoint/{args.run_name}/ckpt-{args.dataset}-{args.model}-{args.seed}-{args.hps.lr}-{args.hps.weight_decay}'

df = pd.read_csv(f"data/{args.dataset}.csv") 
required_cols = ['split', 'target_idx', 'group_idx', 'normalized_weight']
assert all([x in df.columns for x in required_cols]), f"Dataset must have a {required_cols} columns has {df.columns}"
print(f"loaded dataset {args.dataset} of size {len(df)}")
if args.summarize.group_idxs:
    group_names = list(args.summarize.group_idxs)
else:
    group_names = df[args.summarize.group].unique().tolist()
run_name = "-".join(group_names)

if args.summarize.group_idxs:
    old_len = len(df)
    df = df[df[args.summarize.group].isin(group_names)].reset_index(
        drop=True
    )
    # Convert 'group' column to a categorical type with specific ordering
    df[args.summarize.group] = pd.Categorical(
        df[args.summarize.group],
        categories=group_names,
        ordered=True,
    )

    # Sort DataFrame based on custom order
    captions_df = df.sort_values(by=args.summarize.group)
    # trainset = Subset(trainset, captions_df.index.tolist())
    logging.info(
        f"Filtered captions from {old_len} to {len(df)} in groups {group_names}"
    )

if args.summarize.class_idxs:
    old_len = len(df)
    df = df[df[args.summarize.class_col].isin(args.summarize.class_idxs)].reset_index(
        drop=True
    )
    print(
        f"keeping class idxs {args.summarize.class_idxs} before {old_len} after {df[args.summarize.group].value_counts()}"
    )
# Data
print('==> Preparing data..')
trainset = CSVDataset(df, args.summarize.group, args.summarize.class_col, split='train')
valset = CSVDataset(df, args.summarize.group, args.summarize.class_col, split='val')
testset = CSVDataset(df, args.summarize.group, args.summarize.class_col, split='test') if df['split'].nunique() > 2 else valset
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.train.batch_size, num_workers=2,  shuffle=False)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=256, shuffle=False, num_workers=1)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=256, shuffle=False, num_workers=1)


# Model
print('==> Building model..')
print(trainset.classes)
net = getattr(models, args.model)(num_classes = len(trainset.classes))
if args.finetune:
    print("...finetuning")
    # freeze all bust last layer
    for name, param in net.named_parameters(): 
        if 'fc' not in name:
            param.requires_grad = False

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    
run = wandb.init(project=args.name, group='train', name=args.run_name, config=flatten_config(args))
# logger = WandbData(run, testset, args, [s[0] for s in testset.samples], incorrect_only=args.incorrect_only)
wandb.summary['train_size'] = len(trainset)

def load_checkpoint(args, net, optimizer):
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if args.checkpoint_name:
        # see if ./checkpoint/{args.checkpoint_name} is a dir
        if os.path.isdir(f'./checkpoint/{args.checkpoint_name}'):
            checkpoint_name = f'./checkpoint/{args.checkpoint_name}/best.pth'
        else:
            checkpoint_name = f'./checkpoint/{args.checkpoint_name}'
    else:
        assert os.path.exists(ckpt_name)
        checkpoint_name = os.path.join(ckpt_name, 'best.pth')
    checkpoint = torch.load(checkpoint_name)

    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['net'].items():
        if 'module' not in k:
            k = 'module.'+k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k]=v
    
    print(f"Loaded checkpoint at epoch {checkpoint['epoch']} from {checkpoint_name}")
    # net.load_state_dict(checkpoint['net'])
    net.load_state_dict(new_state_dict)
    optimizer.load_state_dict(checkpoint['optim'])

    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return net, optimizer, best_acc, start_epoch

# print("Weights: ", trainset.class_weights)
# criterion = nn.CrossEntropyLoss(weight=torch.tensor(trainset.class_weights).to(device))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.hps.lr,
                      momentum=0.9, weight_decay=args.hps.weight_decay)

if args.hps.lr_scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
elif args.hps.lr_scheduler == 'constant':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                            milestones=[100], # List of epoch indices
                            gamma =0.1) # Multiplicative factor of learning rate decay
elif args.hps.lr_scheduler == 'custom':
    scheduler0 = torch.optim.lr_scheduler.LinearLR(optimizer, 
                     start_factor = 0.008, # The number we multiply learning rate in the first epoch
                     total_iters = 4,) # The number of iterations that multiplicative factor reaches to 1
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                            milestones=[30, 60, 80], # List of epoch indices
                            gamma =0.1) # Multiplicative factor of learning rate decay
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler0, scheduler1])
elif args.hps.lr_scheduler == 'finetune':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs//2, 3*args.epochs//4], gamma=0.1)
else:
    raise ValueError("Unknown scheduler")

if args.resume or (args.eval_only and args.checkpoint_name):
    net, optimizer, best_acc, start_epoch = load_checkpoint(args, net, optimizer)

if args.train.crt:
    assert args.train.loss_weights, "Must specify loss weights for crt"
    assert args.checkpoint_name, "Must specify checkpoint name for crt"
    net, optimizer, best_acc, _ = load_checkpoint(args, net, optimizer)
    start_epoch = 0
    print("...using crt reweighting")

    # Assuming that we want to use the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Check if the model is an instance of DataParallel
    if isinstance(net, nn.DataParallel):
        # Access the underlying model to modify it
        original_model = net.module
    else:
        original_model = net

    # Move the model to the specified device before making changes
    original_model.to(device)

    # reinit last layer
    original_model.fc = nn.Linear(original_model.fc.in_features, len(trainset.classes)).to(device)
    
    # freeze all but last layer
    for name, param in original_model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    # If the model was originally a DataParallel instance, re-wrap the modified model
    # Ensure the original_model has all parameters on the right device before wrapping
    if isinstance(net, nn.DataParallel):
        net = nn.DataParallel(original_model).to(device)
    else:
        net = original_model
    
    ckpt_name += '-crt'
    if not os.path.exists(ckpt_name):
        os.makedirs(ckpt_name)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, groups, filenames) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    wandb.log({'train loss': train_loss/(batch_idx+1), 'train acc': 100.*correct/total, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"]})


def test(epoch, loader, phase='val'):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    all_targets, all_predictions, all_groups, all_filenames = np.array([]), np.array([]), np.array([]), np.array([])
    with torch.no_grad():
        for (inputs, targets, groups, filename) in tqdm(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
        
            try:
                loss = criterion(outputs, targets)
                test_loss += loss.item()
            except:
                print(targets)
                raise ValueError("Loss is nan")
            _, predicted = outputs.max(1)

            all_targets = np.append(all_targets, targets.cpu().numpy())
            all_predictions = np.append(all_predictions, predicted.cpu().numpy())
            all_groups = np.append(all_groups, groups.cpu().numpy())
            all_filenames = np.append(all_filenames, np.array(filename))

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # get per class and per group accuracies
        acc, class_balanced_acc, class_acc, group_acc = evaluate(all_predictions, all_targets, all_groups)
        metrics = {"epoch": epoch, f'{phase} acc': 100.*correct/total, f'{phase} accuracy': acc, 
                   f"{phase} class accuracy": class_acc, f"{phase} balanced accuracy": class_balanced_acc,  f"{phase} balanced group accuracy": group_acc.mean(),
                   **{f"{phase} {loader.dataset.group_names[i]} acc": group_acc[i] for i in range(len(group_acc))}}
        wandb.log(metrics)

    # Save checkpoint.
    # this is changed from the paper, I think checkpointing on acc leads to better results
    acc = 100.*correct/total 
    if acc > best_acc:
        if not args.eval_only or phase == 'val':
            print(f'Saving with acc {acc}..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optim': optimizer.state_dict(),
            }
            
            if not os.path.exists(ckpt_name):
                os.makedirs(ckpt_name)

            # if args.checkpoint_name:
            #     if not os.path.exists(f'./checkpoint/{args.checkpoint_name}'):
            #         os.makedirs(f'./checkpoint/{args.checkpoint_name}')
            #     torch.save(state, f'./checkpoint/{args.checkpoint_name}/best.pth')
            #     wandb.save(f'./checkpoint/{args.checkpoint_name}/best.pth')
            # else:
            torch.save(state, f'./{ckpt_name}/best.pth')
            wandb.save(f'./{ckpt_name}/best.pth')
        best_acc = acc
        wandb.summary['best epoch'] = epoch
        wandb.summary['best val acc'] = best_acc
        wandb.summary['best group acc'] = group_acc
        wandb.summary['best balanced acc'] = class_balanced_acc
        wandb.summary['best class acc'] = class_acc
    if not args.eval_only and epoch % 10 == 0:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optim': optimizer.state_dict()
        }
        # if args.checkpoint_name:
        #     torch.save(state, f'./checkpoint/{args.checkpoint_name}/epoch-{epoch}.pth')
        #     wandb.save(f'./checkpoint/{args.checkpoint_name}/epoch-{epoch}.pth')
        # else:
        torch.save(state, f'./{ckpt_name}/epoch-{epoch}.pth')
        wandb.save(f'./{ckpt_name}/epoch-{epoch}.pth')
    print(np.unique(all_groups, return_counts=True))
    print(loader.dataset.group_names)
    correct = all_targets == all_predictions
    if args.eval_only:
        pred_df = {
            "path": all_filenames,
            "target": all_targets,
            "pred": all_predictions,
            "correct": ['correct' if c else 'incorrect' for c in correct],
            "group": all_groups,
            "group_name": [loader.dataset.group_names[int(g)] for g in all_groups],
            "target_name": [loader.dataset.class_names[int(c)] for c in all_targets],
            "pred_name": [loader.dataset.class_names[int(p)] for p in all_predictions]
        }
        if not os.path.exists(f"predictions/{args.dataset}/{args.run_name}"):
            os.makedirs(f"predictions/{args.dataset}/{args.run_name}")
        pd.DataFrame(pred_df).to_csv(f"predictions/{args.dataset}/{args.run_name}/{args.model}-{args.seed}-{phase}.csv", index=False)

if args.eval_only:
    test(start_epoch, trainloader, phase='train_eval')
    test(start_epoch, valloader, phase='val')
    test(start_epoch, testloader, phase='test')
else:
    for epoch in range(start_epoch, args.epochs):
        train(epoch)
        test(epoch, valloader, phase='val')
        scheduler.step()
        if epoch % 10 == 0:
            test(epoch, testloader, phase='test')
    # load the best checkpoint
    print('==> Loading best checkpoint..')
    net, optimizer, best_acc, start_epoch = load_checkpoint(args, net, optimizer)
    test(epoch, testloader, phase='test')