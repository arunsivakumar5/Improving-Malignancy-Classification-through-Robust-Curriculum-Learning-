
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



def train_erm(params,trainDataloader,validDataloader,model,num_epochs=None,mode='erm'):

    max_val_acc = 0

    criterion = nn.CrossEntropyLoss()
    
    if params['opt'] =='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr =params['learning_rate'],weight_decay =params['w_d']) 
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr =params['learning_rate'],weight_decay =params['w_d']) 
        
    if params['scheduler_choice'] == 1:
        scheduler = init_scheduler({'class_args': {'patience':params['patience'],'factor': params['factor'],'mode':'max'},'class_name': 'ReduceLROnPlateau'},optimizer)   
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

                    acc,a1,a2,a3,a4,a5 = evaluate(validDataloader,model,5,verbose = True)
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
                            torch.save(model.state_dict(), 'C://Users//ASIVAKUM//Desktop//Best_model_erm.pth')
                        elif mode=='cur_erm':
                            torch.save(model.state_dict(), 'C://Users//ASIVAKUM//Desktop//Best_model_cur_erm.pth')
                        elif mode=='random_feature_ext':
                            torch.save(model.state_dict(), 'C://Users//ASIVAKUM//Desktop//Best_model_rand1.pth')
                        elif mode=='Cur_feature_ext':
                            torch.save(model.state_dict(), 'C://Users//ASIVAKUM//Desktop//Best_model_cur1.pth')
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



def train_final_dro(params,model, train_dataloader, val_dataloader,test_dataloader, use_cuda = False, robust=False, num_epochs = 0,stable= True, size_adjustment = None,w=None,m =None):
    
    
    device = torch.device("cuda")
    model = model.to(device)
    
    
    subclass_counts = trainDataset.get_class_counts('subclass')
    
    print('subclass_counts',subclass_counts)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion = LossComputer(criterion, robust,5, subclass_counts, 0.01, stable, 12, False, size_adjustment, use_cuda= use_cuda)
    
    
    
    if params['opt'] =='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr =params['learning_rate'],weight_decay =params['w_d']) 
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr =params['learning_rate'],weight_decay =params['w_d']) 
        
    if params['scheduler_choice'] == 1:
        scheduler = init_scheduler({'class_args': {'patience':params['patience'],'factor': params['factor'],'mode':'min'},'class_name': 'ReduceLROnPlateau'},optimizer) 
    else:
        scheduler = None
    
      
    
    max_val_acc = -1
    for epoch in range(num_epochs):
        
        
        if params['scheduler_choice'] == 1:
            scheduler.last_epoch = epoch - 1
        else:
            pass
        
    
        
        
  
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            
           
            mode = 'traditional'
            model.train()
            
            inputs = inputs.to(device)
            
            loss_targets = targets['superclass']
            loss_targets_cur = targets['subclass']
            loss_targets = loss_targets.to(device)
            loss_targets_cur = loss_targets_cur.to(device)
            logits = model(inputs)
            logits = logits.to(device)
            co = criterion(logits, loss_targets,loss_targets_cur,mode,w=None,epoch=epoch)
            
            
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
        over_val_acc,vacc1,vacc2,vacc3,vacc4,v5 = evaluate(val_dataloader,model, 5)
        valacc = min(vacc1,vacc2,vacc3,vacc4,v5)
        print("epoch", epoch,"Validation Accuracy",min(vacc1,vacc2,vacc3,vacc4,v5))
        if valacc > max_val_acc:
            max_val_acc = valacc
            model = cur_model
            old_model = model
            if mode=='gDRO':
                torch.save(model.state_dict(), 'C://Users//ASIVAKUM//Desktop//Best_model_gdro.pth')
            elif mode=='cur_gDRO':
                torch.save(model.state_dict(), 'C://Users//ASIVAKUM//Desktop//Best_model_cur_gdro.pth')
            elif mode=='random_gDRO':
                torch.save(model.state_dict(), 'C://Users//ASIVAKUM//Desktop//Best_model_rand2.pth')
            elif mode=='Cur_gDRO':
                torch.save(model.state_dict(), 'C://Users//ASIVAKUM//Desktop//Best_model_cur2.pth')
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