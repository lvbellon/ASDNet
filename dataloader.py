import os
import json
from tkinter import Image
import nibabel
import random
import numpy as np
import cv2

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F


# Dataset class 2D
class ABIDE(Dataset):
    def __init__(self, split, gamma=False, transform=None, target_transform=None):
        super(Dataset, self).__init__()

        with open('data/data.json') as json_file:
            data = json.load(json_file)
        with open('data/annotation.json') as json_file:
            annotation = json.load(json_file)
        
        '''with open('data/data_example_6.json') as json_file:
            data = json.load(json_file)
        with open('data/annotation_example_6.json') as json_file:
            annotation = json.load(json_file)'''

        self.data_split = data
        self.annot_split = annotation
        self.split = split
        self.transform = transform
        self.targetTransform = target_transform
        self.gamma = gamma

        self.data_path = []
        self.index = []

        for i in range(len(self.data_split[split])):
            imgsets_file = os.path.join(self.data_split[split][i]['img'])
            self.data_path.append(imgsets_file)
        for i in range(len(self.annot_split[split])):
            self.index.append(self.annot_split[split][i]['annot'])
        
        '''input_image = []
        max = []
        for i in range(len(self.data_path)):
            input_image = nibabel.load(self.data_path[i]).get_fdata()
            breakpoint()
            max.append(np.max(input_image))
        max_val = np.max(max)
        breakpoint()'''

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        target = self.index[index]
        image = nibabel.load(self.data_path[index]).get_fdata()

        img = image[:, :, :, 0]
        time = []
        for i in range(1, image.shape[-1]):
            time.append(i)
        time = np.array(time)
        mean = np.mean(time)
        img = image[:, :, :, int(mean)]
        img = img[:,:,random.randint(25, 35)]
        
        img_final = np.ndarray((61, 73, 3))
        img_final[:, :, 0] = img
        img_final[:, :, 1] = img
        img_final[:, :, 2] = img
        
        if self.gamma:
            img_final = self.gammaCorrection((255 * img_final).astype(np.uint8), 1.5)
            img_final = (img_final.astype(np.float)) / 255
        
        # Image normalization
        #img_final = img_final/14851.370074828148

        if self.transform is not None:
            img_final = self.transform(img_final).float()
        
        if self.targetTransform is not None:
            target = self.targetTransform(target)
        
        return img_final, target
    
    def gammaCorrection(self, src, gamma):
        invGamma = 1 / gamma
    
        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
 
        return cv2.LUT(src, table)
    
    def apply_ct_window(img, window):
        # window = (window width, window level)
        R = (img-window[1]+0.5*window[0])/window[0]
        R[R<0] = 0
        R[R>1] = 1
        return R

#breakpoint()
#dataset = ABIDE(split='train', gamma=True)
#print(dataset[9])
#print(len(dataset))

# Dataset class 3D
class ABIDE3D(Dataset):
    def __init__(self, split, transform=None, target_transform=None):
        super(Dataset, self).__init__()

        with open('data/data.json') as json_file:
            data = json.load(json_file)
        with open('data/annotation.json') as json_file:
            annotation = json.load(json_file)
        
        '''with open('data/data_example_6.json') as json_file:
            data = json.load(json_file)
        with open('data/annotation_example_6.json') as json_file:
            annotation = json.load(json_file)'''

        self.data_split = data
        self.annot_split = annotation
        self.split = split
        self.transform = transform
        self.targetTransform = target_transform

        self.data_path = []
        self.index = []

        for i in range(len(self.data_split[split])):
            imgsets_file = os.path.join(self.data_split[split][i]['img'])
            self.data_path.append(imgsets_file)
        for i in range(len(self.annot_split[split])):
            self.index.append(self.annot_split[split][i]['annot'])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        target = self.index[index]
        image = nibabel.load(self.data_path[index]).get_fdata()    

        img = image[:, :, :, 0]
        time = []
        for i in range(1, image.shape[-1]):
            time.append(i)
        time = np.array(time)
        mean = np.mean(time)
        img = image[:, :, :, int(mean)]
        img = img[:,:,8:]
        
        img = cv2.resize(img, (61,61), interpolation=cv2.INTER_LINEAR)

        # Image normalization
        #img = img/14851.370074828148

        if self.transform is not None:
            img = self.transform(img)

        if self.targetTransform is not None:
            target = self.targetTransform(target)

        return img, target


#breakpoint()
#dataset = ABIDE3D(split='valid')
#print(dataset[9])
# print(len(dataset))