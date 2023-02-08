import scipy.stats as st
import numpy as np


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

import models




def Average(lst):
    return sum(lst) / len(lst)

def report_CI(temp_list):
    """ Computes confidence interval for a list of accuracies """
    return st.t.interval(alpha=0.95, df=len(temp_list)-1, loc=np.mean(temp_list), scale=st.sem(temp_list)) 


def evaluate(dataloader, model, num_subclasses, verbose=False):
    """
    Evaluate the model's accuracy and subclass sensitivities
    :param dataloader: The dataloader for the validation/testing data
    :param model: The model to evaluate
    :param num_subclasses: The number of subclasses to evaluate on, this should be equal to the number of subclasses present in the data
    :param verbose: Whether to print the results
    :return: A tuple containing the overall accuracy and the sensitivity for each subclass
    """
    model.eval()

    num_samples = np.zeros(num_subclasses)
    subgroup_correct = np.zeros(num_subclasses)
    with torch.no_grad():
        X = dataloader.dataset.features.cuda()
        y = dataloader.dataset.labels.cuda()
        c = dataloader.dataset.subclasses.cuda()

        pred = model(X)

        for subclass in range(num_subclasses):
            subclass_idx = c == subclass
            num_samples[subclass] += torch.sum(subclass_idx)
            subgroup_correct[subclass] += (pred[subclass_idx].argmax(1) == y[subclass_idx]).type(
                torch.float).sum().item()

    subgroup_accuracy = subgroup_correct / num_samples
    
    accuracy = sum(subgroup_correct) / sum(num_samples)

    if verbose:
        print("Accuracy:", accuracy, "\nAccuracy over subgroups:", subgroup_accuracy, "\nWorst Group Accuracy:",
              min(subgroup_accuracy))

    return (accuracy, *subgroup_accuracy)



def get_train_splits(file='./data/LIDC_3_4_Ratings_wMSE.csv',val_prop=0.10,test_prop=0.3):
    
    ''' Randomizes the train-validation-Test instances with fixed proportion'''
    
    for i in range(1,31):
        
        df_splits = pd.read_csv(file, index_col=0)
        df_splits.reset_index(inplace=True,drop = True)

        df_splits.drop('splits',inplace=True,axis=1)
        prop = test_prop

        df1 = df_splits[df_splits['Malignancy'] == 1 ]
        df2 = df_splits[df_splits['Malignancy'] == 2 ]
        df3 = df_splits[df_splits['Malignancy'] == 3 ]
        df4 = df_splits[df_splits['Malignancy'] == 4 ]
        df5 = df_splits[df_splits['Malignancy'] == 5 ]

        
        df_sample = df1.sample(frac=1)
        train_size1 = int(test_prop* len(df_sample))
        

        df_split1_1 = df_sample[:train_size1]
        df_split1_2 = df_sample[train_size1:]
        

            
        df_sample2 = df2.sample(frac=1)
        train_size2 = int(test_prop* len(df_sample2))

        df_split2_1 = df_sample2[:train_size2]
        df_split2_2 = df_sample2[train_size2:]

  
            
        df_sample3 = df3.sample(frac=1)
        train_size3 = int(test_prop* len(df_sample3))

        df_split3_1 = df_sample3[:train_size3]
        df_split3_2 = df_sample3[train_size3:]
        
        

        df_sample4 = df4.sample(frac=1)
        train_size4 = int(test_prop* len(df_sample4))

        df_split4_1 = df_sample4[:train_size4]
        df_split4_2 = df_sample4[train_size4:]
        
      

        df_sample5 = df5.sample(frac=1)
        train_size5 = int(test_prop* len(df_sample5))

        df_split5_1 = df_sample5[:train_size5]
        df_split5_2 = df_sample5[train_size5:]

        df_split12 = pd.concat([df_split1_1, df_split2_1], axis=0)
        df_split34 = pd.concat([df_split3_1, df_split4_1], axis=0)
        df_split125 = pd.concat([df_split12, df_split5_1], axis=0)
        df_split12345_test = pd.concat([df_split125, df_split34], axis=0)

        df_split12 = pd.concat([df_split1_2, df_split2_2], axis=0)
        df_split34 = pd.concat([df_split3_2, df_split4_2], axis=0)
        df_split125 = pd.concat([df_split12, df_split5_2], axis=0)
        df_split12345_valtrain = pd.concat([df_split125, df_split34], axis=0)


        df_split12345_test['splits'] = 2

        df_splits = df_split12345_valtrain
        prop= val_prop

        df1 = df_splits[df_splits['Malignancy'] == 1 ]
        df2 = df_splits[df_splits['Malignancy'] == 2 ]
        df3 = df_splits[df_splits['Malignancy'] == 3 ]
        df4 = df_splits[df_splits['Malignancy'] == 4 ]
        df5 = df_splits[df_splits['Malignancy'] == 5 ]

        df_sample = df1.sample(frac=1)
        
        train_size1 = int(val_prop* len(df_sample))

        df_split1_1 = df_sample[:train_size1]
        df_split1_2 = df_sample[train_size1:]

        df_sample2 = df2.sample(frac=1)
        train_size2 = int(val_prop* len(df_sample2))

        df_split2_1 = df_sample2[:train_size2]
        df_split2_2 = df_sample2[train_size2:]


        df_sample3 = df3.sample(frac=1)
        train_size3 = int(val_prop* len(df_sample3))

        df_split3_1 = df_sample3[:train_size3]
        df_split3_2 = df_sample3[train_size3:]

        df_sample4 = df4.sample(frac=1)
        train_size4 =int(val_prop* len(df_sample4))

        df_split4_1 = df_sample4[:train_size4]
        df_split4_2 = df_sample4[train_size4:]

        df_sample5 = df5.sample(frac=1)
        train_size5 = int(val_prop* len(df_sample5))

        df_split5_1 = df_sample5[:train_size5]
        df_split5_2 = df_sample5[train_size5:]

        df_split12 = pd.concat([df_split1_1, df_split2_1], axis=0)
        df_split34 = pd.concat([df_split3_1, df_split4_1], axis=0)
        df_split125 = pd.concat([df_split12, df_split5_1], axis=0)
        df_split12345_val = pd.concat([df_split125, df_split34], axis=0)

        df_split12345_val['splits'] = 1

        df_split12 = pd.concat([df_split1_2, df_split2_2], axis=0)
        df_split34 = pd.concat([df_split3_2, df_split4_2], axis=0)
        df_split125 = pd.concat([df_split12, df_split5_2], axis=0)
        df_split12345_train = pd.concat([df_split125, df_split34], axis=0)

        df_split12345_train['splits'] = 0



        df_split12345_train.sort_values('noduleID', inplace=True)
        df_split12345_train.reset_index(drop=True, inplace=True)

        df_split12345_val.sort_values('noduleID', inplace=True)
        df_split12345_val.reset_index(drop=True, inplace=True)

        df_split12345_test.sort_values('noduleID', inplace=True)
        df_split12345_test.reset_index(drop=True, inplace=True)

        df_trains = pd.concat([df_split12345_train, df_split12345_val], axis=0)
        df_all = pd.concat([df_trains, df_split12345_test], axis=0)
        
        df_all.sort_values('noduleID', inplace=True)
        df_all.reset_index(drop=True, inplace=True)
        
        
        
        save_at = os.path.join('./data/Train_splits/nodule_split_', str(i) ,'.csv').replace("\\","")

        df_all.to_csv(save_at)
    
    
def random_split(prop,split_file):
    
    prop = prop/100
    
    
    

    df_splits = pd.read_csv(split_file)
    
    df_splits = df_splits[df_splits['splits'] <= 1]
    
    
    df1 = df_splits[df_splits['Malignancy'] == 1 ]
    df2 = df_splits[df_splits['Malignancy'] == 2 ]
    df3 = df_splits[df_splits['Malignancy'] == 3 ]
    df4 = df_splits[df_splits['Malignancy'] == 4 ]
    df5 = df_splits[df_splits['Malignancy'] == 5 ]
    
    df_sample = df1.sample(frac=1)
    
    print("splitting data into",int(prop*100),":",int((1-prop)*100),"splits")
    train_size1 = int(prop* len(df_sample))
    
    df_split1_1 = df_sample[:train_size1]
    df_split1_2 = df_sample[train_size1:]
    
    df_sample2 = df2.sample(frac=1)
    train_size2 = int(prop* len(df_sample2))

    df_split2_1 = df_sample2[:train_size2]
    df_split2_2 = df_sample2[train_size2:]
    

    
    df_sample3 = df3.sample(frac=1)
    train_size3 = int(prop* len(df_sample3))
    
    df_split3_1 = df_sample3[:train_size3]
    df_split3_2 = df_sample3[train_size3:]
    
    df_sample4 = df4.sample(frac=1)
    train_size4 = int(prop* len(df_sample4))
    
    df_split4_1 = df_sample4[:train_size4]
    df_split4_2 = df_sample4[train_size4:]
    
    df_sample5 = df5.sample(frac=1)
    train_size5 = int(prop* len(df_sample5))
    
    df_split5_1 = df_sample5[:train_size5]
    df_split5_2 = df_sample5[train_size5:]
    
    
    df_split11 = pd.concat([df_split1_1, df_split2_1], axis=0)
    df_split22 = pd.concat([df_split1_2, df_split2_2], axis=0)
    df_split11 = pd.concat([df_split11,df_split5_2], axis=0)
    df_split22 = pd.concat([df_split22,df_split5_1], axis=0)
    
    df_split01  = pd.concat([df_split3_1, df_split4_1], axis=0)
    df_split1 = pd.concat([df_split11, df_split01], axis=0)
    
    df_split02  = pd.concat([df_split3_2, df_split4_2], axis=0)
    df_split2 = pd.concat([df_split22, df_split02], axis=0)

    df_split1.sort_values('noduleID', inplace=True)
    df_split2.sort_values('noduleID', inplace=True)
    
    
    df_split1 = (df_split1.iloc[: , 1:]).reset_index(drop = True)
    df_split2 = (df_split2.iloc[: , 1:]).reset_index(drop = True)

    
    return df_split1,df_split2


def init_scheduler(scheduler_config, optimizer):
    
        scheduler_class = getattr(schedulers, scheduler_config['class_name'])
        return scheduler_class(optimizer, **scheduler_config['class_args'])