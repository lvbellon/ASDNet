#! /usr/bin/env python
import os
import copy
import json
import time
from tkinter import N
import monai
import logging
import tqdm
import random
import argparse
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from skimage.morphology import ball


from itertools import cycle
from sklearn.metrics import (precision_recall_curve, average_precision_score, PrecisionRecallDisplay, 
                            confusion_matrix, f1_score, recall_score, precision_score, plot_confusion_matrix)

from utils import Evaluator
from loss import *

import torch
import torchio as tio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

from dataloader import ABIDE, ABIDE3D  # NOQA
import ASDNet3D.model.ASDNet3D as ASDNet3D

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/test_model_monai_without_pre-training_BCE_6layers')

# Arguments
parser = argparse.ArgumentParser(description='Autism classification')
parser.add_argument('--mode', type=str, default='test', choices=['test', 'demo'],
                    help='type of test to calculate the results')
parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'resnet101'],
                    help ='model classification for training (default: resnet50)')
parser.add_argument('--pretrained-model', action='store_true', default=True,
                    help = 'model classification with pretrained weights for training')   
parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101'],
                    help="backbone model for model classification")
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--n-class', type=int, default=2,
                    help='input number of classes for training (default:2)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
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
parser.add_argument('--gamma', type=float, default=2, metavar='M', choices=[2, 0.5],
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
parser.add_argument('--save', type=str, default='model_monai_without_pre-training_BCE_6layers.pt',
                    help='file on which to save model weights')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

mean = (0.0)
std = (1.0)
# Data loaders
testDataset = ABIDE3D(split='test', 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ]),
        target_transform=None
        )

test_loader = DataLoader(testDataset, args.batch_size, shuffle=False, **kwargs)


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
    criterion = nn.CrossEntropyLoss(reduction=args.reduction)
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

out = 'test'

if not osp.exists(out):
    os.mkdir(out)

if not osp.exists(out+'/'+args.save.split('.')[0]):
    os.mkdir(out+'/'+args.save.split('.')[0])


def test(epoch):
    model.eval()
    test_loss = 0.
    metrics.reset()
    lbl_trues, lbl_preds = [], []
    trues, preds = [], []
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)      
        data = data.unsqueeze(1).float()
        target = nn.functional.one_hot(target, num_classes=args.n_class).to(torch.float32)

        with torch.no_grad():
            output = model(data)

        test_loss += criterion(activation(output.squeeze(1)), target).item()
        pred = output.data.cpu().float()
        pred = F.softmax(pred, dim=1).numpy()
        lbl = target.data.max(1)[1].cpu().numpy()
        target = target.data.cpu().numpy()
        predic = np.argmax(pred, axis=1)
        lbl_trues.append(lbl)
        lbl_preds.append(predic)
        trues.append(target)
        preds.append(pred)

    
       # Compute precision-recall curve
        prec = dict()
        rec = dict()
        plt.figure()
        _, ax = plt.subplots()
        colors =  cycle(['navy', 'turquoise'])
        for i, color in zip(range(args.n_class), colors):
            lt = [item[i] for item in trues[0]]
            lp = [item[i] for item in preds[0]]
            prec[i], rec[i], _ = precision_recall_curve(lt, lp)
            display = PrecisionRecallDisplay(recall=rec[i], precision=prec[i], average_precision=average_precision_score(lt, lp), estimator_name=None)
            display.plot(ax=ax, name=f"Precision-recall curve for class {i}", color=color)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall curve")
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(out,args.save.split('.')[0],'PR_curve.png'))

    acc, acc_cls = metrics.label_accuracy_score(lbl, predic, n_class=args.n_class)
    conf_matrix = metrics._generate_matrix(lbl_trues[0], lbl_preds[0])
    tn, fp, fn, tp = confusion_matrix(lbl_trues[0], lbl_preds[0]).ravel()
    f_measure = f1_score(lbl_trues[0], lbl_preds[0])
    precision = precision_score(lbl_trues[0], lbl_preds[0])
    recall = recall_score(lbl_trues[0], lbl_preds[0])

    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test", acc, epoch)

    print('Validation:')
    print('[Epoch: %d, numImages: %5d]' % (epoch, len(test_loader.dataset)))
    print('Accuracy {:<10.6f} Mean Accuracy {:<10.6f}'.format(acc, acc_cls))
    print('F measure {:<10.6f} Precision {:<10.6f} Recall {:<10.6f} '.format(f_measure, precision, recall))
    print(f'Confusion matrix \n{conf_matrix}')
    print('tn {:<10.6f} fp {:<10.6f} fn {:<10.6f} tp {:<10.6f}'.format(tn, fp, fn, tp))
    print('Loss: %.3f' % test_loss)
    return test_loss


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed
       by 10 at every specified step
       Adapted from PyTorch Imagenet example:
       https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def visualization():
    if not osp.exists(out+'/'+args.save.split('.')[0]+'/visualization'):
        os.mkdir(out+'/'+args.save.split('.')[0]+'/visualization')
    
    
    with open('data/data.json') as json_file:
        data = json.load(json_file)

    
    test_path = []
    for i in range(len(data['test'])):
        imgsets_file = os.path.join(data['test'][i]['img'])
        test_path.append(imgsets_file)
    
    id = random.randint(0, len(test_path))
    
    data, target = testDataset.__getitem__(id)
    if args.cuda:
        data, target = data.cuda(), torch.tensor(target)
    data, target = Variable(data), Variable(target)
    target = nn.functional.one_hot(target, num_classes=args.n_class).to(torch.float32)
    with torch.no_grad():
        output = model(data.unsqueeze(0).unsqueeze(1).cuda().float())
    
    pred = output.data.cpu().float()
    pred = F.softmax(pred, dim=1).numpy()
    
    target = target.data.max(0)[1].cpu().numpy()
    pred = np.argmax(pred, axis=1)
    data = torch.permute(data,(1,2,0))

    def show_slices(slices):
        """ Function to display row of image slices """
        fig, axes = plt.subplots(1, len(slices))
        for i, slices in enumerate(slices):
            axes[i].imshow(slices.T, cmap="gray", origin="lower")
            axes[i].axis('off')

    slice_0 = data.cpu().numpy()[26, :, :]
    slice_1 = data.cpu().numpy()[:, 30, :]
    slice_2 = data.cpu().numpy()[:, :, 26]
    show_slices([slice_0, slice_1, slice_2])
    plt.tight_layout()
    plt.suptitle(f'Predicted Class: {int(pred)} and true class: {int(target)}')
    plt.savefig(os.path.join(out+'/'+args.save.split('.')[0]+'/visualization'+'/cualitative_results.png'))


if __name__ == '__main__':
    best_loss = None
    if load_model:
        best_loss = test(0)
        if args.mode == 'demo':
            visualization()
    try:
        for epoch in tqdm.trange(1, args.epochs + 1):
            epoch_start_time = time.time()
            test_loss = test(epoch)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s '.format(
                epoch, time.time() - epoch_start_time))
            print('-' * 89)

            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
                with open(args.save, 'wb') as fp:
                    state = model.state_dict()
                    torch.save(state, fp)
            else:
                adjust_learning_rate(optimizer, args.gamma, epoch)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

writer.flush()