#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
niftidataset.transforms
transformations to apply to images in dataset
Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Oct 24, 2018
"""

# Taken from https://github.com/perone/medicaltorch/blob/master/medicaltorch/transforms.py

import skimage
import numpy as np
import numbers
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class NormalizeInstance3D(MTTransform):
    """Normalize a tensor volume with mean and standard deviation estimated
    from the sample itself.
    :param mean: mean value.
    :param std: standard deviation value.
    """
    def __call__(self, input_data):
        mean, std = input_data.mean(), input_data.std()

        if mean != 0 or std != 0:
            input_data_normalized = F.normalize(input_data,
                                    [mean for _ in range(0,input_data.shape[0])],
                                    [std for _ in range(0,input_data.shape[0])])

        return input_data_normalized

