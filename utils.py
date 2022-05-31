import numpy as np
import glob
import os
import re
from typing import Optional, Tuple

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pydicom.pixel_data_handlers import apply_voi_lut
from torch import Tensor

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def _generate_matrix(self, gt_image, gt_pred):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        confusion_matrix = np.bincount(
            self.num_class * gt_image[mask].astype(int) +
            gt_pred[mask], minlength=self.num_class ** 2).reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, gt_pred):
        self.confusion_matrix += self._generate_matrix(gt_image.flatten(), gt_pred.flatten())

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
    
    def fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def label_accuracy_score(self, label_trues, label_preds, n_class):
        hist = np.zeros((n_class, n_class))
        for lt, lp in zip(label_trues, label_preds):
            hist += self.fast_hist(lt.flatten(), lp.flatten(), n_class)
        acc = np.diag(hist).sum() / hist.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        return acc, acc_cls
