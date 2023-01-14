import argparse

import matplotlib.pyplot as plt
import numpy as np 

import torch 
import torchvision 
import torchvision.transforms as transforms 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torch.optim.lr_scheduler as schedulers
import sklearn 
from sklearn.metrics import accuracy_score

import pickle
import os

import torch.nn.functional as F

from sklearn import metrics

from functools import partial

import tarfile

import random
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
from torchvision import datasets, models, transforms


import logging
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch.nn as nn
import math

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


import torch.optim as optimizers
import torch.optim.lr_scheduler as schedulers
import os


import torch


import pandas as pd
import os
import argparse
from math import ceil


class LIDC_Dataset(Dataset):
    def __init__(self,features,labels,subclasses):
        '''
        INPUTS:
        features: list of features (as Pytorch tensors)
        labels:   list of corresponding lables
        subclasses: list of corresponding subclasses
        '''

        self.X = features
        self.features = features
        self.labels = labels
        self.subclasses = subclasses

    
        yList = []
 
        for i in labels:
            if i == 0:
                yList.append(0)
            elif i ==1:
                yList.append(1)
            else:
                yList.append(2)
        subYList = []
        print("subclasses",subclasses)
        for j in subclasses:
            print(j)
            if j == 0:
                subYList.append(0)
            elif j ==1:
                subYList.append(1)
            elif j == 2:
                subYList.append(2)
            elif j == 3:
                subYList.append(3)
            elif j == 4:
                subYList.append(4)
                
        self.Y_dict = {}
        self.Y_dict['superclass'] = torch.tensor(yList)
        self.Y_dict['subclass'] = torch.tensor(subYList)
                
                
                
    def __getitem__(self, index):
        
        return self.X[index], {'superclass':self.Y_dict['superclass'][index], 'subclass':self.Y_dict['subclass'][index]}
        
    
    def __len__(self):
        return len(self.X)
    
    def get_class_counts(self, className):
        return torch.tensor(np.bincount(np.array(self.Y_dict[className])))