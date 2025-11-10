#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VGG19 Loss Module using torchvision pre-trained VGG19.
This module extracts features from intermediate layers of VGG19 and computes an L1 loss
between the features of the input and target images.
Images are assumed to be in range [0, 1] and will be normalized using ImageNet statistics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG19FeatureExtractor(nn.Module):
    def __init__(self, layers=('relu2_2', 'relu3_4', 'relu4_4', 'relu5_4'), use_input_norm=True):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True).features
        self.use_input_norm = use_input_norm
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        
        self.layer_name_mapping = {
            'relu1_1': 1,
            'relu1_2': 3,
            'relu2_1': 6,
            'relu2_2': 8,
            'relu3_1': 11,
            'relu3_2': 13,
            'relu3_3': 15,
            'relu3_4': 17,
            'relu4_1': 20,
            'relu4_2': 22,
            'relu4_3': 24,
            'relu4_4': 26,
            'relu5_1': 29,
            'relu5_2': 31,
            'relu5_3': 33,
            'relu5_4': 35,
        }
        
        self.selected_layers = {name: idx for name, idx in self.layer_name_mapping.items() if name in layers}
        self.vgg = vgg19
        
        # Freeze VGG19 parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        features = {}
        output = x
        idx_to_name = {idx: name for name, idx in self.selected_layers.items()}
        for i, layer in enumerate(self.vgg):
            output = layer(output)
            if i in idx_to_name:
                features[idx_to_name[i]] = output
        return features



