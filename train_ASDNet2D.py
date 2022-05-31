#! /usr/bin/env python
import copy
import time
from tkinter import N
import tqdm
import argparse
import numpy as np
import os.path as osp
from PIL import Image
from distutils.version import LooseVersion
from sklearn.metrics import (f1_score, recall_score, precision_score, confusion_matrix)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torchvision.models.segmentation as segmentation
from torch.autograd import Variable

from dataloader import ABIDE  # NOQA
from model.ResNet import (ResNet, resnet50, resnet101)
from utils import Evaluator
from loss import *

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/model_resnet50_with_pre-training_ultimo')

# Arguments
parser = argparse.ArgumentParser(description='Autism classification')
parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'resnet101'],
                    help ='model classification for training (default: resnet50)')
parser.add_argument('--pretrained-model', action='store_true', default=True,
                    help = 'model classification with pretrained weights for training')   
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--n-class', type=int, default=2,
                    help='input number of classes for training (default:2)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'],
                    help='optimizer for training (default: Adam)')
parser.add_argument('--momentum', type=float, default=0.99, choices=[0.5, 0.99], metavar='M',
                    help='optimizer momentum (default: 0.5)')
parser.add_argument('--weight_decay', type=float, default=0.0002, choices=[0.0005, 0.0002, 0.0001],
                    help='optimizer weight decay (default: 0.0005)')
parser.add_argument('--alpha', type=float, default=-1, metavar='M', 
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument('--gamma', type=float, default=0.5, metavar='M', choices=[2, 0.5],
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument('--loss', type=str, default='BCE', choices = ['CE', 'BCE', 'FocalLoss'],
                    help='model loss (default: CE)')          
parser.add_argument('--act-loss', action='store_true', default=True,
                    help = 'function activation loss')  
parser.add_argument('--reduction', type=str, default='mean', choices = ['mean', 'sum'],
                    help='reduction to apply to the loss (default: mean)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
parser.add_argument('--save', type=str, default='model_resnet50_with_pre-training_ultimo.pt',
                    help='file on which to save model weights')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# Data loaders
trainDataset = ABIDE(split='train', gamma=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ]),
        target_transform=None
        )

train_loader = DataLoader(trainDataset, args.batch_size, shuffle=True, **kwargs)

valDataset = ABIDE(split='valid', gamma=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ]),
        target_transform=None
        )

val_loader = DataLoader(valDataset, args.test_batch_size, shuffle=False, **kwargs)


# Model
if args.model == 'resnet50':
    if args.pretrained_model == True: 
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(2048, args.n_class)
    else:
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(2048, args.n_class)
    
elif args.model == 'resnet101':
    if args.pretrained_model == True: 
        model = resnet101(pretrained=True)
        model.fc = nn.Linear(2048, args.n_class)
    else:
        model = resnet101(pretrained=False)
        model.fc = nn.Linear(2048, args.n_class)

cp_model = copy.deepcopy(model)

if args.cuda:
    model.cuda()

load_model = False
if osp.exists(args.save):
    with open(args.save, 'rb') as fp:
        state = torch.load(fp)
        model.load_state_dict(state)
        load_model = True

# Optimizer & loss
if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.loss == 'CE':
    criterion = nn.CrossEntropyLoss(reduction=args.reduction, ignore_index=-1)
elif args.loss == 'BCE':
    if args.act_loss == True:
        activation = nn.Sigmoid()
        criterion = nn.BCEWithLogitsLoss(reduction=args.reduction)
    else:
        criterion = nn.BCEWithLogitsLoss(reduction=args.reduction)
elif args.loss == 'FocalLoss':
    if args.act_loss == True:
        activation = nn.Sigmoid()
        criterion = FocalLoss()
    else:
        criterion = FocalLoss()

metrics = Evaluator(args.n_class)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target).flatten()
        data = torch.permute(data,(0,1,2,3))
        target = nn.functional.one_hot(target, num_classes=args.n_class).to(torch.float32)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(activation(output.squeeze(1)), target)
        pred = output.data.cpu().float()
        pred = F.softmax(pred, dim=1).numpy()
        pred = np.argmax(pred, axis=1)
        target = target.data.max(1)[1].cpu().numpy()
        acc, acc_cls = metrics.label_accuracy_score(target, pred, n_class=args.n_class)
        print('Accuracy {:<10.6f} Mean Accuracy {:<10.6f}'.format(acc, acc_cls))
        #loss = criterion(activation(output.squeeze()), target)
        #loss = CB_loss(output.squeeze(), target.float(), [3,1], args.n_class, "sigmoid", 0.9999)
        #loss = sigmoid_focal_loss(output, target, alpha=args.alpha, gamma=args.gamma, reduction=args.reduction)
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Accuracy/train', acc, epoch)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def valid(epoch):
    model.eval()
    metrics.reset()
    val_loss = 0.
    lbl_trues, lbl_preds = [], []
    for data, target in val_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target).flatten()
        data = torch.permute(data,(0,1,2,3))       
        target = nn.functional.one_hot(target, num_classes=args.n_class).to(torch.float32)

        with torch.no_grad():
            output = model(data)
        val_loss += criterion(activation(output.squeeze(1)), target).item()
        #test_loss += CB_loss(output.squeeze(), target.float(), [3,1], args.n_class, "sigmoid", 0.9999)
        #test_loss += sigmoid_focal_loss(output, target, alpha=args.alpha, gamma=args.gamma, reduction=args.reduction)
        pred = output.data.cpu().float()
        pred = F.softmax(pred, dim=1).numpy()
        pred = np.argmax(pred, axis=1)
        target = target.data.max(1)[1].cpu().numpy()
        #pred = F.softmax(pred, dim=0).numpy()
        #pred = np.argmax(pred, axis=0)
        lbl_preds.append(pred)
        lbl_trues.append(target)

    acc, acc_cls = metrics.label_accuracy_score(target, pred, n_class=args.n_class)
    conf_matrix = metrics._generate_matrix(lbl_trues[0], lbl_preds[0])
    tn, fp, fn, tp = confusion_matrix(lbl_trues[0], lbl_preds[0]).ravel()
    f_measure = f1_score(lbl_trues[0], lbl_preds[0])
    precision = precision_score(lbl_trues[0], lbl_preds[0])
    recall = recall_score(lbl_trues[0], lbl_preds[0])

    writer.add_scalar("Loss/valid", val_loss, epoch)
    writer.add_scalar("Accuracy/valid", acc, epoch)

    print('Validation:')
    print('[Epoch: %d, numImages: %5d]' % (epoch, len(val_loader.dataset)))
    print('Accuracy {:<10.6f} Mean Accuracy {:<10.6f}'.format(acc, acc_cls))
    print('F measure {:<10.6f} Precision {:<10.6f} Recall {:<10.6f} '.format(f_measure, precision, recall))
    print(f'Confusion matrix \n{conf_matrix}')
    print('tn {:<10.6f} fp {:<10.6f} fn {:<10.6f} tp {:<10.6f}'.format(tn, fp, fn, tp))
    print('Loss: %.3f' % val_loss)
    return val_loss


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed
       by 10 at every specified step
       Adapted from PyTorch Imagenet example:
       https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    best_loss = None
    if load_model:
        best_loss = valid(0)
    try:
        for epoch in tqdm.trange(1, args.epochs + 1):
            epoch_start_time = time.time()
            train(epoch)
            val_loss = valid(epoch)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s '.format(
                epoch, time.time() - epoch_start_time))
            print('-' * 89)

            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                with open(args.save, 'wb') as fp:
                    state = model.state_dict()
                    torch.save(state, fp)
            else:
                adjust_learning_rate(optimizer, args.gamma, epoch)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

writer.flush()