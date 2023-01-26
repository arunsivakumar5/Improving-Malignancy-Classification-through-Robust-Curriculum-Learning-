
import matplotlib.pyplot as plt  # for plotting
import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
import torch.optim.lr_scheduler as schedulers
import sklearn 
from sklearn.metrics import accuracy_score
from loss import LossComputer
import pickle
from torchvision import datasets, models, transforms

import torch.nn.functional as F

from sklearn import metrics

from functools import partial

import random
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle


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

import utils.image_data_utils as im_utils
import utils.data_utils as d_utils

def train_erm(params,trainDataloader,validDataloader,model,num_epochs=None,mode='erm'):

    max_val_acc = 0

    criterion = nn.CrossEntropyLoss()
    
    if params['opt'] =='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr =params['learning_rate'],weight_decay =params['w_d']) 
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr =params['learning_rate'],weight_decay =params['w_d']) 
        
    if params['scheduler_choice'] == 1:
        scheduler = d_utils.init_scheduler({'class_args': {'patience':params['patience'],'factor': params['factor'],'mode':'max'},'class_name': 'ReduceLROnPlateau'},optimizer)   
    else:
        scheduler = None

    EPOCHS=num_epochs
    device = torch.device("cuda")

    for epoch_num in range(EPOCHS):
            if scheduler:
                scheduler.last_epoch = epoch_num - 1
            else:
                pass
            
            
            model.train()
            for train_input,train_label in trainDataloader:


                train_label = train_label['superclass']
                
                train_label = train_label.to(device)
                train_input = train_input.to(device)
                
                optimizer.zero_grad()
                output = model(train_input)
                
                _, predictions = output.max(1)
      
                loss = criterion(output,train_label)
                batch_loss =loss
                

                
                
                model.zero_grad()
                
                batch_loss.backward()
                optimizer.step()
                
            
                   
                    
            model.eval()
            cur_model = model
            
            with torch.no_grad():

                    acc,a1,a2,a3,a4,a5 = d_utils.evaluate(validDataloader,model,5,verbose = True)
                    if scheduler:
                        scheduler.step(acc) 
                    else:
                        pass
                    
                    print("acc",acc)
                    print("Max acc",max_val_acc)
                    if acc > max_val_acc:
                        max_val_acc =acc
                        model = cur_model
                        old_model = model
                        if mode=='erm':
                            torch.save(model.state_dict(), './models/Best_model_erm.pth')
                        elif mode=='cur_erm':
                            torch.save(model.state_dict(), './models/Best_model_cur_erm.pth')
                        elif mode=='random_feature_ext':
                            torch.save(model.state_dict(), './models/Best_model_rand1.pth')
                        elif mode=='Cur_feature_ext':
                            torch.save(model.state_dict(), './models/Best_model_cur1.pth')
                        else:
                            print("Model weights unsaved")
                            pass
                        perfect_epoch = epoch_num
                        print("perfect epoch",perfect_epoch)
                    else:
                        try:
                            model = old_model
                        except:
                            old_model = model
                    
                   
                        
          
            
    return model,max_val_acc



def train_gdro(params,model, train_dataloader, val_dataloader, use_cuda = True, robust=True, num_epochs = 0,stable= True, size_adjustment = None,mode =None,subclass_counts=None):
    
    
    device = torch.device("cuda")
    model = model.to(device)
    
    
    
    
    print('subclass_counts',subclass_counts)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion = LossComputer(criterion, robust,5, subclass_counts, 0.01, stable, 12, False, size_adjustment, use_cuda= use_cuda)
    
    
    
    if params['opt'] =='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr =params['learning_rate'],weight_decay =params['w_d']) 
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr =params['learning_rate'],weight_decay =params['w_d']) 
        
    if params['scheduler_choice'] == 1:
        scheduler =  d_utils.init_scheduler({'class_args': {'patience':params['patience'],'factor': params['factor'],'mode':'max'},'class_name': 'ReduceLROnPlateau'},optimizer) 
    else:
        scheduler = None
    
      
    
    max_val_acc = -1
    for epoch in range(num_epochs):
        
        
        if params['scheduler_choice'] == 1:
            scheduler.last_epoch = epoch - 1
        else:
            pass
        
    
        
        
  
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            
           
  
            model.train()
            
            inputs = inputs.to(device)
            
            loss_targets = targets['superclass']
            loss_targets_cur = targets['subclass']
            loss_targets = loss_targets.to(device)
            loss_targets_cur = loss_targets_cur.to(device)
            logits = model(inputs)
            logits = logits.to(device)
            co = criterion(logits, loss_targets,loss_targets_cur)
            
            
            loss, (losses, corrects) = co
            
            optimizer.zero_grad()
            
            loss.backward()
            predicteds = logits.argmax(1)
            actuals = targets['superclass']
            actuals = actuals.to(device)
            
            trains_accs = (predicteds == actuals).sum()/predicteds.shape[0]
            
            optimizer.step()
            
        
        
            
            
        
            
            
        model.eval() 
        cur_model = model       
        over_val_acc,vacc1,vacc2,vacc3,vacc4,v5 = d_utils.evaluate(val_dataloader,model, 5)
        valacc = min(vacc1,vacc2,vacc3,vacc4,v5)
        print("epoch", epoch,"Validation Accuracy",min(vacc1,vacc2,vacc3,vacc4,v5))
        if valacc > max_val_acc:
            max_val_acc = valacc
            model = cur_model
            old_model = model
            if mode=='gDRO':
                try:
                    torch.save(model.state_dict(), './models/Best_model_gdro.pth')
                    path = './models/Best_model_gdro.pth'
                except:
                    os.makedirs(path)
            elif mode=='cur_gDRO':
                try:
                    torch.save(model.state_dict(), './models/Best_model_cur_gdro.pth')
                    path = './models/Best_model_cur_gdro.pth'
                except:
                    os.makedirs(path)
            elif mode=='random_gDRO':
                try:
                    torch.save(model.state_dict(), './models/Best_model_rand2.pth')
                except:
                    pass
            elif mode=='Cur_gDRO':
                torch.save(model.state_dict(), './models/Best_model_cur2.pth')
            else:
                print("Model weights unsaved")
                pass
            perfect_epoch = epoch
            print("perfect epoch",perfect_epoch)
                
            
        else:
            try:
                model = old_model
            except:
                old_model = model
                
        if params['scheduler_choice'] == 1:
            scheduler.step(valacc)
        else:
            pass
        
                
                

    return model,max_val_acc




def train_gdro_ct(params,model, train_dataloader1, val_dataloader1,train_dataloader2,val_dataloader2,num_epochs = 0,mode =None, subclass_counts1=None,subclass_counts2=None, use_cuda = True, robust=True, stable= True, size_adjustment = None):
    
    
    device = torch.device("cuda")
    
    model_new = torchvision.models.resnet18(pretrained=True).to(device)
    num_ftrs = model_new.fc.in_features
    model_new.fc = nn.Linear(num_ftrs, 2)
    model = model_new
    
    model = model.to(device)
    
    
    
    if params['opt'] =='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr =params['learning_rate'],weight_decay =params['w_d']) 
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr =params['learning_rate'],weight_decay =params['w_d']) 
        
    if params['scheduler_choice'] == 1:
        scheduler =  d_utils.init_scheduler({'class_args': {'patience':params['patience'],'factor': params['factor'],'mode':'max'},'class_name': 'ReduceLROnPlateau'},optimizer) 
    else:
        scheduler = None
    
      
    
    max_val_acc = -1
    for epoch in range(num_epochs):
        
        
            if params['scheduler_choice'] == 1:
                scheduler.last_epoch = epoch - 1
            else:
                pass
        
    
            
            
            
            model.train()
            if epoch <75:
                criterion = torch.nn.CrossEntropyLoss(reduction='none')
                criterion = LossComputer(criterion, robust,5, subclass_counts1, 0.01, stable, 12, False, size_adjustment, use_cuda= use_cuda)
                for batch_idx, (inputs, targets) in enumerate(train_dataloader1):
                                 
            
                    inputs = inputs.to(device)
            
                    loss_targets = targets['superclass']
                    loss_targets_cur = targets['subclass']
                    loss_targets = loss_targets.to(device)
                    loss_targets_cur = loss_targets_cur.to(device)
                    if epoch ==1:
                        print(model)
                    else:
                        pass
                    logits = model(inputs)
                    logits = logits.to(device)
                    co = criterion(logits, loss_targets,loss_targets_cur)
            
            
                    loss, (losses, corrects) = co
            
                    optimizer.zero_grad()
            
                    loss.backward()
                    predicteds = logits.argmax(1)
                    actuals = targets['superclass']
                    actuals = actuals.to(device)
            
                    trains_accs = (predicteds == actuals).sum()/predicteds.shape[0]
            
                    optimizer.step()

            else:
                criterion = torch.nn.CrossEntropyLoss(reduction='none')
                criterion = LossComputer(criterion, robust,5, subclass_counts2, 0.01, stable, 12, False, size_adjustment, use_cuda= use_cuda)
                for batch_idx, (inputs, targets) in enumerate(train_dataloader2):
                                 
            
                    inputs = inputs.to(device)
            
                    loss_targets = targets['superclass']
                    loss_targets_cur = targets['subclass']
                    loss_targets = loss_targets.to(device)
                    loss_targets_cur = loss_targets_cur.to(device)
                    if epoch==75:
                        model_new = model
                        num_ftrs = model_new.fc.in_features
                        model_new.fc = nn.Linear(num_ftrs, 3)
                        model = model_new
                        model = model.to(device)   
                    else:
                        pass
 
    
                    logits = model(inputs)
                    logits = logits.to(device)
                    co = criterion(logits, loss_targets,loss_targets_cur)
            
            
                    loss, (losses, corrects) = co
            
                    optimizer.zero_grad()
            
                    loss.backward()
                    predicteds = logits.argmax(1)
                    actuals = targets['superclass']
                    actuals = actuals.to(device)
            
                    trains_accs = (predicteds == actuals).sum()/predicteds.shape[0]
            
                    optimizer.step()
            
        
        
            
            
        
            
            
                model.eval() 
                cur_model = model      
                if epoch < 75:
                    over_val_acc,vacc1,vacc2,vacc3,vacc4= d_utils.evaluate(valloader,model, 4)
                else:
                    over_val_acc,vacc1,vacc2,vacc3,vacc4,v5 = d_utils.evaluate(valloader,model, 5)
                
                
                try:
                    valacc = min(vacc1,vacc2,vacc3,vacc4)
                    print("epoch", epoch,"Validation Accuracy",min(vacc1,vacc2,vacc3,vacc4))
                except:
                    valacc = min(vacc1,vacc2,vacc3,vacc4,v5)
                    print("epoch", epoch,"Validation Accuracy",min(vacc1,vacc2,vacc3,vacc4,v5))
                
                if epoch>74:

                    if valacc > max_val_acc:
                        max_val_acc = valacc
                        model = cur_model
                        old_model = model
                        if mode=='gDRO':
                            try:
                                torch.save(model.state_dict(), './models/Best_model_gdro.pth')
                                path = './models/Best_model_gdro.pth'
                            except:
                                os.makedirs(path)
                        elif mode=='cur_gDRO':
                            try:
                                torch.save(model.state_dict(), './models/Best_model_cur_gdro.pth')
                                path = './models/Best_model_cur_gdro.pth'
                            except:
                                os.makedirs(path)
                        elif mode=='random_gDRO':
                            try:
                                torch.save(model.state_dict(), './models/Best_model_rand2.pth')
                            except:
                                pass
                        elif mode=='Cur_gDRO':
                            torch.save(model.state_dict(), './models/Best_model_cur2.pth')
                        else:
                            print("Model weights unsaved")
                            pass
                        perfect_epoch = epoch
                        print("perfect epoch",perfect_epoch)
                
            
                    else:
                        model = old_model
                    
                    
                    if params['scheduler_choice'] == 1:
                        scheduler.step(valacc)
                    else:
                        pass
                else:
                    if params['scheduler_choice'] == 1:
                        scheduler.step(valacc)
                    else:
                        pass
                
                
     
    return model,max_val_acc