import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import wandb
import os
from tqdm import tqdm
import clip
import torchvision.transforms as transforms
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
import torch
import torchvision

class Dataset(torch.utils.data.Dataset):
    def __init__(self, group1, group2, transform=None):
        self.group1 = [(g['path'], 0) for g in group1]
        self.group2 = [(g['path'], 1) for g in group2]
        self.samples = self.group1 + self.group2
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target

def create_model(name="CLIP", fine_tune=False):
    if name == "CLIP":
        model, transform = clip.load("RN50", device="cuda")
    elif name == "ResNet18":
        model = torchvision.models.resnet18(pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    model.fc = nn.Linear(512, 2)
    if fine_tune:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    model = model.cuda()
    return model, transform

def create_dataset(group1, group2, transform=None):
    return Dataset(group1, group2, transform=transform)

def train(model, train_loader, optimizer, epoch):
    # train for an epoch and return loss and accuracy
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    logging.info(f'Train Epoch: {epoch} \tLoss: {train_loss:.6f} \tAccuracy: {correct / len(train_loader.dataset):.6f}')
    wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": correct / len(train_loader.dataset)})
    return train_loss, correct / len(train_loader.dataset)
       

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for target, data in test_loader:
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    return test_loss, correct / len(test_loader.dataset)

def classifier_sampler(group1, group2, model="RN50", fine_tune=True):
    # create model and trainset
    model, transform = create_model(model, fine_tune)
    trainset = create_dataset(group1, group2, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    # train model
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(20):
        train(model, train_loader, optimizer, epoch)
    # save model
    torch.save(model.state_dict(), "classifier.pt")
    return model