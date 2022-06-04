#! /usr/bin/env python
import logging
import copy
import time
from tkinter import N
import tqdm
import argparse
import numpy as np
import os.path as osp
from sklearn.metrics import (f1_score, recall_score, precision_score, confusion_matrix)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable


from dataloader import ABIDE3D  # NOQA
import model.ASDNet3D as  ASDNet3D
from utils import Evaluator
from loss import *

import transforms as mt_transforms


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/model_monai_without_pre-training_softmax_resnet101_3blocks_7layers')

# Arguments
parser = argparse.ArgumentParser(description='Autism classification')
parser.add_argument('--model', type=str, default='resnet101', choices=['resnet50', 'resnet101'],
                    help ='model classification for training (default: resnet50)')
parser.add_argument('--pretrained-model', action='store_true', default=False,
                    help = 'model classification with pretrained weights for training')   
parser.add_argument('--batch-size', type=int, default=12, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--n-class', type=int, default=2,
                    help='input number of classes for training (default:2)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'Adadelta'],
                    help='optimizer for training (default: Adam)')
parser.add_argument('--momentum', type=float, default=0.99, choices=[0.5, 0.99], metavar='M',
                    help='optimizer momentum (default: 0.5)')
parser.add_argument('--weight_decay', type=float, default=0.0002, choices=[0.0005, 0.0002, 0.0001],
                    help='optimizer weight decay (default: 0.0005)')
parser.add_argument('--alpha', type=float, default=-1, metavar='M', 
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument('--gamma', type=float, default=0.5, metavar='M', choices=[2, 0.5],
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument('--loss', type=str, default='FocalLoss', choices = ['CE', 'BCE', 'FocalLoss'],
                    help='model loss (default: CE)') 
parser.add_argument('--loss_type', type=str, default='softmax', choices = ['focal', 'sigmoid', 'softmax'],
                    help='model loss (default: CE)')
parser.add_argument('--weights', type=list, default=[1, 0.5], metavar='M',
                    help='weights for model training')             
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
parser.add_argument('--save', type=str, default='model_monai_without_pre-training_softmax_resnet101_3blocks_7layers.pt',
                    help='file on which to save model weights')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}


# Data loaders
trainDataset = ABIDE3D(split='train',
        transform=transforms.Compose([
            transforms.ToTensor(),
            mt_transforms.NormalizeInstance3D()
            ]),
        target_transform=None
        )

train_loader = DataLoader(trainDataset, args.batch_size, shuffle=True, **kwargs)

valDataset = ABIDE3D(split='valid', 
        transform=transforms.Compose([
            transforms.ToTensor(),
            mt_transforms.NormalizeInstance3D()
            ]),
        target_transform=None
        )

val_loader = DataLoader(valDataset, args.test_batch_size, shuffle=False, **kwargs)


# Model
if args.model == 'resnet50':
    model = ASDNet3D.resnet50(spatial_dims=3, n_input_channels=1, num_classes=args.n_class)
    if args.pretrained_model == True:
        model_dict = model.state_dict()
        pretrain = torch.load('/home/lvbellon/PROJECT/pretrain/resnet_50_23dataset.pth')
        pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
        missing = tuple({k for k in model_dict.keys() if k not in pretrain['state_dict']})
        logging.debug(f"missing in pretrained: {len(missing)}")
        inside = tuple({k for k in pretrain['state_dict'] if k in model_dict.keys()})
        logging.debug(f"inside pretrained: {len(inside)}")
        unused = tuple({k for k in pretrain['state_dict'] if k not in model_dict.keys()})
        pretrain['state_dict'] = {k: v for k, v in pretrain['state_dict'].items() if k in model_dict.keys()}
        model.load_state_dict(pretrain['state_dict'], strict=False)
elif args.model == 'resnet101':
    model = ASDNet3D.resnet101(spatial_dims=3, n_input_channels=1, num_classes=args.n_class)
    if args.pretrained_model == True:
        model_dict = model.state_dict()
        pretrain = torch.load('/home/lvbellon/PROJECT/pretrain/resnet_50_23dataset.pth')
        pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
        missing = tuple({k for k in model_dict.keys() if k not in pretrain['state_dict']})
        logging.debug(f"missing in pretrained: {len(missing)}")
        inside = tuple({k for k in pretrain['state_dict'] if k in model_dict.keys()})
        logging.debug(f"inside pretrained: {len(inside)}")
        unused = tuple({k for k in pretrain['state_dict'] if k not in model_dict.keys()})
        pretrain['state_dict'] = {k: v for k, v in pretrain['state_dict'].items() if k in model_dict.keys()}
        model.load_state_dict(pretrain['state_dict'], strict=False)

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
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas = (0.9, 0.99))
elif args.optimizer == 'Adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06)

if args.loss == 'CE':
    criterion = nn.CrossEntropyLoss(reduction=args.reduction, ignore_index=-1)
elif args.loss == 'BCE':
    if args.act_loss == True:
        activation = nn.Sigmoid()
        #criterion = nn.BCEWithLogitsLoss(reduction=args.reduction)
        criterion = nn.BCELoss(reduction=args.reduction)
    else:
        #criterion = nn.BCEWithLogitsLoss(reduction=args.reduction)
        criterion = nn.BCELoss(reduction=args.reduction)
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
        data, target = Variable(data), Variable(target)
        data = data.unsqueeze(1).float()
        target = nn.functional.one_hot(target, num_classes=args.n_class).to(torch.float32)
        optimizer.zero_grad()
        output = model(data)
        loss = CB_loss(target.float(), output.squeeze(), args.weights, args.n_class, args.loss_type, 0.9999, args.gamma)
        pred = output.data.cpu().float()
        pred = F.softmax(pred, dim=1).numpy()
        pred = np.argmax(pred, axis=1)
        target = target.data.max(1)[1].cpu().numpy()
        acc, acc_cls = metrics.label_accuracy_score(target, pred, n_class=args.n_class)
        print('Accuracy {:<10.6f} Mean Accuracy {:<10.6f}'.format(acc, acc_cls))
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
    val_loss = 0.
    metrics.reset()
    lbl_trues, lbl_preds = [], []
    for data, target in val_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)      
        data = data.unsqueeze(1).float()
        target = nn.functional.one_hot(target, num_classes=args.n_class).to(torch.float32)

        with torch.no_grad():
            output = model(data)

        val_loss += CB_loss(target.float(), output, args.weights, args.n_class, args.loss_type, 0.9999, args.gamma)
        pred = output.data.cpu().float()
        pred = F.softmax(pred, dim=1).numpy()
        pred = np.argmax(pred, axis=1)
        target = target.data.max(1)[1].cpu().numpy()
        lbl_trues.append(target)
        lbl_preds.append(pred)
    
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