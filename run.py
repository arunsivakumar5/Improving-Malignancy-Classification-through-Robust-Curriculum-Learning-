import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

import argparse

import matplotlib.pyplot as plt
import numpy as np 
import pingouin as pg
from scipy import stats
from IPython.display import display
import torch 
import torchvision 
import torchvision.transforms as transforms 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torch.optim.lr_scheduler as schedulers
import sklearn 
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, BatchSampler,SequentialSampler
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









import utils.image_data_utils as im_utils
import utils.data_utils as d_utils
import models
from dataset import LIDC_Dataset
from loss import LossComputer
from train import train_erm,train_gdro,train_gdro_ct,train_erm_ct, train_gdro_ct_five,train_gdro_new,train_gdro_ct_new





val_acc_base = []
val_acc_final = []
train_acc_erm = []

B_major_acc = []
B_minor_acc = []
M_minor_acc = []
M_major_acc = []

B_maj_list_new01 = []
B_min_list_new01 = []
M_min_list_new01 = []
M_maj_list_new01 = []
test_acc_list01 = []



acc_lst = []

acc_lst1 = []

acc_lst2 = []

acc_lst3 = []

acc_lst4 = []

file_num = []

acc_list_new1 = []

all_spec_ls = []
all_sens_ls = []

over_acc_cur_gdro_lst = []
cur_gdro1_lst = []
cur_gdro2_lst = []
cur_gdro3_lst = []
cur_gdro4_lst = []
cur_gdro5_lst = []




over_acc_erm_lst_cur = []
erm1_lst_cur = []
erm2_lst_cur = []
erm3_lst_cur = []
erm4_lst_cur = []
erm5_lst_cur = []


acc_list = []
acc_list2 = []


over_acc_erm_lst= []
erm1_lst = []
erm2_lst = []
erm3_lst = []
erm4_lst = []
erm5_lst = []

acc_list_train =[]
acc_lists = []

over_acc_gdro_lst = []
gdro1_lst = []
gdro2_lst = []
gdro3_lst = []
gdro4_lst = []
gdro5_lst = []



over_acc_cris_rand = []
cris_rand_1 = []
cris_rand_2  = []
cris_rand_3  = []
cris_rand_4  = []


over_acc_cris_rep = []
cris_rep_1 = []
cris_rep_2  = []
cris_rep_3  = []
cris_rep_4  = []


from io import StringIO,BytesIO 



DEVICE = torch.device('cuda')






parser = argparse.ArgumentParser()


parser.add_argument('--method', '-n', default='ERM', help='The method using which the classifier is trained ')
parser.add_argument('--trials',  type=int, default=30, help='The number of times we repeat the experiment with different train-validation-test splits ')
parser.add_argument('--epochs',  '-t', default=50,  help='The total number of epochs the deep learning model must be trained')
parser.add_argument('--threshold_epoch',  type=int, default=2,  help='The epoch at which we change from 2-class problem to 3-class problem')
parser.add_argument('--freeze', action='store',  help='If all layers except the classifier layer are to be frozen to carry out transfer learning or to just use the pretrained weights as initial weights')
parser.add_argument('--curriculum', action='store',  help='If curriculum information has to be used to sort the instances by Easy to hard as data is fed into the classiifer sequentially')
parser.add_argument('--significance', action='store',  help='If significance tests should be carried out between the results of a classifier with curriculum learning and the same classifier without curriculum learning')


args = parser.parse_args()


method = args.method





DEVICE =  torch.device('cuda')





if method =='ERM':
    
    
            
     
    if args.curriculum == 'Both':
        
        params ={'learning_rate': 0.0005,
                                'patience':2,
                                'batch_size': 128,  
                                'w_d': 0.005,
                                'factor': 0.2,
                                'scheduler_choice':1,
                                'opt': 'Adam' }

        for i in range(1,args.trials + 1): 

            print("file",i)
            
            split_file = os.path.join('./data/splits/nodule_split_?.csv').replace("?",str(i))
            
            data_easy,datas_hard = im_utils.get_cur_features(device=DEVICE,file=split_file,mode='unsorted') 

            datas_cur = im_utils.get_erm_features(device=DEVICE,file=split_file,mode='curriculum') 

            _,cv_data,test_data = datas_cur
            
            train_data_easy,_ = data_easy
            train_data_hard,_ = datas_hard

            trainDataset1 = LIDC_Dataset(*train_data_easy)
            trainDataset2 = LIDC_Dataset(*train_data_hard)
         

            tr = trainDataset1
            train_dataloader1 = DataLoader(tr, batch_size =params['batch_size'],shuffle=False,sampler=torch.utils.data.RandomSampler(tr,replacement=True)) 
            
            

            tr = trainDataset2           
            train_dataloader2 = DataLoader(tr, batch_size =params['batch_size'],shuffle=False,sampler=torch.utils.data.RandomSampler(tr,replacement=True))


            validDataset = LIDC_Dataset(*cv_data)
            testDataset = LIDC_Dataset(*test_data)


            
            val = validDataset
            val_weights =   im_utils.get_sampler_weights(validDataset.subclasses)

            test =testDataset
            test_weights =   im_utils.get_sampler_weights(testDataset.subclasses)

            
            val_dataloader = DataLoader(val,batch_size = len(validDataset) ,shuffle = False,sampler = torch.utils.data.WeightedRandomSampler(val_weights,len(val_weights)) )
            test_dataloader = DataLoader(test, batch_size = len(testDataset) ,shuffle = False,sampler=torch.utils.data.WeightedRandomSampler(test_weights,len(test_weights)) )  
            
            

            device = torch.device('cuda')

            if args.freeze == 'Yes':
                model = models.TransferModel18(num_classes=2)
                model2 = models.TransferModel18(num_classes=2)  #Initially 2 for matching states and changed later.
            else:
                model = models.TransferModel18(freeze=False,num_classes=2)
                model2 = models.TransferModel18(freeze=False,num_classes=2)

            steps1 = math.ceil(len(trainDataset1) / params['batch_size'])
            steps2 = math.ceil(len(trainDataset2) / params['batch_size'])
            
            modelA,max_acc = train_erm_ct(params,train_dataloader1,val_dataloader,train_dataloader2,model,num_epochs=50,mode='cur_erm',steps1=steps1,steps2=steps2)
            modelA.load_state_dict(torch.load('.//models//Best_model_cur_erm.pth'))
            print("Cur ERM trained!")
      
            over_acc_erm,erm1,erm2,erm3,erm4,erm5 = d_utils.evaluate(test_dataloader,modelA, 5,verbose = True)
            
            over_acc_erm_lst_cur.append(over_acc_erm)
            erm1_lst_cur.append(erm1)
            erm2_lst_cur.append(erm2)
            erm3_lst_cur.append(erm3)
            erm4_lst_cur.append(erm4)
            erm5_lst_cur.append(erm5)
        
            itemlist =over_acc_erm_lst_cur
            with open('./test_results/over_test_acc_erm_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = erm1_lst_cur
            with open('./test_results/acc1_erm_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = erm2_lst_cur
            with open('./test_results/acc2_erm_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = erm3_lst_cur
            with open('./test_results/acc3_erm_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = erm4_lst_cur
            with open('./test_results/acc4_erm_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = erm5_lst_cur
            with open('./test_results/acc5_erm_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            datas = im_utils.get_erm_features(device=DEVICE,file=split_file,mode='traditional')

            train_data,cv_data,test_data = datas

            trainDataset = LIDC_Dataset(*train_data)
                  

            tr = trainDataset           
            train_dataloader = DataLoader(tr, batch_size =params['batch_size'],shuffle=False,sampler=torch.utils.data.RandomSampler(tr,replacement=True))
            
            

            device = torch.device('cuda')

            if args.freeze == 'Yes':
                model = models.TransferModel18(num_classes=3)
            else:
                model = models.TransferModel18(freeze=False,num_classes=3)

            
            steps = math.ceil(len(trainDataset) / params['batch_size'])
            modelA,max_acc = train_erm(params,train_dataloader,val_dataloader,model,num_epochs=50,mode='erm',steps=steps)
            modelA.load_state_dict(torch.load('.//models//Best_model_erm.pth'))
            print("Traditional ERM trained!")

            over_acc_erm,erm1,erm2,erm3,erm4,erm5 = d_utils.evaluate(test_dataloader,modelA, 5,verbose = True)

            over_acc_erm_lst.append(over_acc_erm)
            erm1_lst.append(erm1)
            erm2_lst.append(erm2)
            erm3_lst.append(erm3)
            erm4_lst.append(erm4)
            erm5_lst.append(erm5)
        
            itemlist =over_acc_erm_lst
            with open('./test_results/over_test_acc_erm.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = erm1_lst
            with open('./test_results/acc1_erm.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = erm2_lst
            with open('./test_results/acc2_erm.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = erm3_lst
            with open('./test_results/acc3_erm.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = erm4_lst
            with open('./test_results/acc4_erm.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)
            itemlist = erm5_lst
            with open('./test_results/acc5_erm.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

        
        
            
    

elif method =='gDRO':

    
    

    if args.curriculum == 'Both':

        for i in range(1,args.trials + 1): 

            print("Trial",i)
            params ={'learning_rate': 0.0005,
                                'patience':2,
                                'batch_size': 128,  
                                'w_d': 0.005,
                                'factor': 0.2,
                                'scheduler_choice':1,
                                'opt': 'Adam' }
            
                

            file_num.append(i)
                
            split_file = os.path.join('./data/Train_splits1/nodule_split_?.csv').replace("?",str(i))
            
            data_easy,datas_hard = im_utils.get_cur_features(device=DEVICE,file=split_file,mode='unsorted')  

            datas_cur = im_utils.get_erm_features(device=DEVICE,file=split_file,mode='curriculum') 

            _,cv_data,test_data = datas_cur
            

            train_data_easy,cv_data_easy = data_easy
            train_data_hard,cv_data_hard = datas_hard

            trainDataset1 = LIDC_Dataset(*train_data_easy)
            validDataset1 = LIDC_Dataset(*cv_data_easy)

            trainDataset2 = LIDC_Dataset(*train_data_hard)
            validDataset2 = LIDC_Dataset(*cv_data_hard)

            testDataset = LIDC_Dataset(*test_data)

            

            tr = trainDataset1
            val = validDataset1
            test=testDataset

            
            subclass_counts1=trainDataset1.get_class_counts('subclass')

            train_weights1 = im_utils.get_sampler_weights(trainDataset1.subclasses)    


            
            train_dataloader1 = DataLoader(tr, batch_size =params['batch_size'],shuffle=False,sampler=torch.utils.data.WeightedRandomSampler(train_weights1,len(train_weights1)) )
            

            try:
                val_weights1 =   im_utils.get_sampler_weights(validDataset1.subclasses)
            except:
                val_weights1 =   im_utils.get_sampler_weights(validDataset1.labels)

            val_dataloader1 = DataLoader(val,batch_size = len(validDataset1) ,shuffle = False,sampler = torch.utils.data.WeightedRandomSampler(val_weights1,len(val_weights1)) )



            tr = trainDataset2
            val = validDataset2

            subclass_counts2=trainDataset2.get_class_counts('subclass')
            train_weights2 = im_utils.get_sampler_weights(trainDataset2.subclasses)    


            
            train_dataloader2 = DataLoader(tr, batch_size =params['batch_size'],shuffle=False,sampler=torch.utils.data.WeightedRandomSampler(train_weights2,len(train_weights2)) )
            

            val_weights2 =   im_utils.get_sampler_weights(validDataset2.subclasses)
            val_dataloader2 = DataLoader(val,batch_size = len(validDataset2) ,shuffle = False,sampler = torch.utils.data.WeightedRandomSampler(val_weights2,len(val_weights2)) )


            validDataset = LIDC_Dataset(*cv_data)
            testDataset = LIDC_Dataset(*test_data)


            
            val = validDataset
            val_weights =   im_utils.get_sampler_weights(validDataset.subclasses)
            test_weights =   im_utils.get_sampler_weights(testDataset.subclasses)

            
            val_dataloader = DataLoader(val,batch_size = len(validDataset) ,shuffle = False,sampler = torch.utils.data.WeightedRandomSampler(val_weights,len(val_weights)) )
            test_dataloader = DataLoader(test, batch_size = len(testDataset) ,shuffle = False,sampler=torch.utils.data.WeightedRandomSampler(test_weights,len(test_weights)) )  
            
            

            device = torch.device('cuda')

            if args.freeze == 'Yes':
                
                model = torchvision.models.resnet18(pretrained=True).to(device)   
                for param in model.parameters():
                    param.requires_grad = False
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 2)
                model_cur = model.to(device)  #Initially 2 for matching states and changed later.

                model = torchvision.models.resnet18(pretrained=True).to(device)   
                for param in model.parameters():
                    param.requires_grad = False
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 3)
                model_gdro = model.to(device)  
            else:
                
                model = torchvision.models.resnet18(pretrained=True).to(device)   
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 2)
                model_cur = model.to(device)  #Initially 2 for matching states and changed later.
            
                model = torchvision.models.resnet18(pretrained=True).to(device)   
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 3)
                model_gdro = model.to(device)  
            

            modelA,max_acc = train_gdro_ct(params,model,train_dataloader1,val_dataloader1,train_dataloader2,val_dataloader2,num_epochs=300,mode='cur_gDRO',subclass_counts1=subclass_counts1,subclass_counts2=subclass_counts2)
            modelA.load_state_dict(torch.load('.//models//Best_model_cur_gdro.pth'))
            print("Cur gDRO trained!")

            over_acc_cur_gdro,cur_gdro1,cur_gdro2,cur_gdro3,cur_gdro4,cur_gdro5 = d_utils.evaluate(test_dataloader,modelA, 5,verbose = True)
      
            over_acc_cur_gdro_lst.append(over_acc_cur_gdro)
            cur_gdro1_lst.append(cur_gdro1)
            cur_gdro2_lst.append(cur_gdro2)
            cur_gdro3_lst.append(cur_gdro3)
            cur_gdro4_lst.append(cur_gdro4)
            cur_gdro5_lst.append(cur_gdro5)

            itemlist =over_acc_cur_gdro_lst
            with open('./test_results/over_test_acc_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = cur_gdro1_lst
            with open('./test_results/acc1_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = cur_gdro2_lst
            with open('./test_results/acc2_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = cur_gdro3_lst
            with open('./test_results/acc3_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = cur_gdro4_lst
            with open('./test_results/acc4_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = cur_gdro5_lst
            with open('./test_results/acc5_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)
            
            itemlist = file_num
            with open('./test_results/test_logs.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)
            
            datas = im_utils.get_erm_features(device=DEVICE,file=split_file,mode='traditional')

            train_data,cv_data,test_data = datas

            trainDataset = LIDC_Dataset(*train_data)
            

            subclass_counts=trainDataset.get_class_counts('subclass')

            tr = trainDataset
            

            train_weights = im_utils.get_sampler_weights(trainDataset.subclasses)    


            sampler = torch.utils.data.WeightedRandomSampler(
                        train_weights,
                        len(train_weights))
            train_dataloader = DataLoader(tr, batch_size =params['batch_size'],sampler=sampler )

            

            device = torch.device('cuda')
            if args.freeze == 'Yes':
                model = models.TransferModel18(num_classes=3)
            else:
                model = models.TransferModel18(freeze=False,num_classes=3)
            
   
            
            modelA,max_acc = train_gdro(params,model,train_dataloader,val_dataloader,num_epochs=300,mode ='gDRO',subclass_counts = subclass_counts)
            modelA.load_state_dict(torch.load('.//models//Best_model_gdro.pth'))
            print("Traditional gDRO trained!")

            over_acc_gdro,gdro1,gdro2,gdro3,gdro4,gdro5 = d_utils.evaluate(test_dataloader,modelA, 5,verbose = True)

            over_acc_gdro_lst.append(over_acc_gdro)
            gdro1_lst.append(gdro1)
            gdro2_lst.append(gdro2)
            gdro3_lst.append(gdro3)
            gdro4_lst.append(gdro4)
            gdro5_lst.append(gdro5)
        
            itemlist =over_acc_gdro_lst
            with open('./test_results/over_test_acc_gdro.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = gdro1_lst
            with open('./test_results/acc1_gdro.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = gdro2_lst
            with open('./test_results/acc2_gdro.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = gdro3_lst
            with open('./test_results/acc3_gdro.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = gdro4_lst
            with open('./test_results/acc4_gdro.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)
            itemlist = gdro5_lst
            with open('./test_results/acc5_gdro.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

    elif args.curriculum == 'Both_verbose':

        for i in range(1,args.trials + 1): 

            print("Trial",i)
            params ={'learning_rate': 0.0005,
                                'patience':2,
                                'batch_size': 128,  
                                'w_d': 0.005,
                                'factor': 0.2,
                                'scheduler_choice':1,
                                'opt': 'Adam' }
            
                

            file_num.append(i)
                
            split_file = os.path.join('./data/splits/nodule_split_?.csv').replace("?",str(i))
            
            data_easy,datas_hard = im_utils.get_cur_features(device=DEVICE,file=split_file,mode='unsorted')  

            datas_cur = im_utils.get_erm_features(device=DEVICE,file=split_file,mode='traditional') 

            _,cv_data,test_data = datas_cur
            

            train_data_easy,cv_data_easy = data_easy
            train_data_hard,cv_data_hard = datas_hard

            trainDataset1 = LIDC_Dataset(*train_data_easy)
            validDataset1 = LIDC_Dataset(*cv_data_easy)

            trainDataset2 = LIDC_Dataset(*train_data_hard)
            validDataset2 = LIDC_Dataset(*cv_data_hard)

            testDataset = LIDC_Dataset(*test_data)

            

            tr = trainDataset1
            val = validDataset1
            test=testDataset

            
            subclass_counts1=trainDataset1.get_class_counts('subclass')

            train_weights1 = im_utils.get_sampler_weights(trainDataset1.subclasses)    


            
            train_dataloader1 = DataLoader(tr, batch_size =params['batch_size'],shuffle=False,sampler=torch.utils.data.WeightedRandomSampler(train_weights1,len(train_weights1)) )
            

            try:
                val_weights1 =   im_utils.get_sampler_weights(validDataset1.subclasses)
            except:
                val_weights1 =   im_utils.get_sampler_weights(validDataset1.labels)

            val_dataloader1 = DataLoader(val,batch_size = len(validDataset1) ,shuffle = False,sampler = torch.utils.data.WeightedRandomSampler(val_weights1,len(val_weights1)) )



            tr = trainDataset2
            val = validDataset2

            subclass_counts2=trainDataset2.get_class_counts('subclass')
            train_weights2 = im_utils.get_sampler_weights(trainDataset2.subclasses)    


            
            train_dataloader2 = DataLoader(tr, batch_size =params['batch_size'],shuffle=False,sampler=torch.utils.data.WeightedRandomSampler(train_weights2,len(train_weights2)) )
            

            val_weights2 =   im_utils.get_sampler_weights(validDataset2.subclasses)
            val_dataloader2 = DataLoader(val,batch_size = len(validDataset2) ,shuffle = False,sampler = torch.utils.data.WeightedRandomSampler(val_weights2,len(val_weights2)) )


            validDataset = LIDC_Dataset(*cv_data)
            testDataset = LIDC_Dataset(*test_data)


            
            val = validDataset
            val_weights =   im_utils.get_sampler_weights(validDataset.subclasses)
            test_weights =   im_utils.get_sampler_weights(testDataset.subclasses)

            
            val_dataloader = DataLoader(val,batch_size = len(validDataset) ,shuffle = False,sampler = torch.utils.data.WeightedRandomSampler(val_weights,len(val_weights)) )
            test_dataloader = DataLoader(test, batch_size = len(testDataset) ,shuffle = False,sampler=torch.utils.data.WeightedRandomSampler(test_weights,len(test_weights)) )  
            
            

            device = torch.device('cuda')

            if args.freeze == 'Yes':
                
                model = torchvision.models.resnet18(pretrained=True).to(device)   
                for param in model.parameters():
                    param.requires_grad = False
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 2)
                model_cur = model.to(device)  #Initially 2 for matching states and changed later.

                model = torchvision.models.resnet18(pretrained=True).to(device)   
                for param in model.parameters():
                    param.requires_grad = False
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 3)
                model_gdro = model.to(device)  
            else:
                
                model = torchvision.models.resnet18(pretrained=True).to(device)   
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 2)
                model_cur = model.to(device)  #Initially 2 for matching states and changed later.
            
                model = torchvision.models.resnet18(pretrained=True).to(device)   
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 3)
                model_gdro = model.to(device) 
            
            steps1 = math.ceil(len(trainDataset1) / params['batch_size'])
            steps2 = math.ceil(len(trainDataset2) / params['batch_size'])

            modelA,max_acc,cur_train,cur_vals,overall_val = train_gdro_ct_new(params,model_cur,train_dataloader1,val_dataloader1,train_dataloader2,val_dataloader2,num_epochs=int(args.epochs),mode='cur_gDRO',subclass_counts1=subclass_counts1,subclass_counts2=subclass_counts2,steps1=steps1,steps2=steps2)
            modelA.load_state_dict(torch.load('.//models//Best_model_cur_gdro.pth'))
            print("Cur gDRO trained!")
            print(file_num)
            over_acc_cur_gdro,cur_gdro1,cur_gdro2,cur_gdro3,cur_gdro4,cur_gdro5 = d_utils.evaluate(test_dataloader,modelA, 5,verbose = True)
      
            over_acc_cur_gdro_lst.append(over_acc_cur_gdro)
            cur_gdro1_lst.append(cur_gdro1)
            cur_gdro2_lst.append(cur_gdro2)
            cur_gdro3_lst.append(cur_gdro3)
            cur_gdro4_lst.append(cur_gdro4)
            cur_gdro5_lst.append(cur_gdro5)

            

            itemlist =over_acc_cur_gdro_lst
            with open('./test_results/over_test_acc_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = cur_gdro1_lst
            with open('./test_results/acc1_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = cur_gdro2_lst
            with open('./test_results/acc2_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = cur_gdro3_lst
            with open('./test_results/acc3_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = cur_gdro4_lst
            with open('./test_results/acc4_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = cur_gdro5_lst
            with open('./test_results/acc5_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)
            
            itemlist = file_num
            with open('./test_results/test_logs.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)
            
            itemlist = cur_train
            with open('./test_results/cur_train_lst.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = cur_vals
            with open('./test_results/cur_val_lst.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = overall_val
            with open('./test_results/overall_val_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            datas = im_utils.get_erm_features(device=DEVICE,file=split_file,mode='traditional')

            train_data,cv_data,test_data = datas

            trainDataset = LIDC_Dataset(*train_data)
            

            subclass_counts=trainDataset.get_class_counts('subclass')

            tr = trainDataset
            

            train_weights = im_utils.get_sampler_weights(trainDataset.subclasses)    


            sampler = torch.utils.data.WeightedRandomSampler(
                        train_weights,
                        len(train_weights))
            train_dataloader = DataLoader(tr, batch_size =params['batch_size'],sampler=sampler )

            

            device = torch.device('cuda')
            if args.freeze == 'Yes':
                model = models.TransferModel18(num_classes=3)
            else:
                model = models.TransferModel18(freeze=False,num_classes=3)
            
            steps = math.ceil(len(trainDataset) / params['batch_size'])
            
            modelB,max_acc,gdro_train,gdro_vals,overall_gdro_vals = train_gdro_new(params,model_gdro,train_dataloader,val_dataloader,num_epochs=int(args.epochs),mode ='gDRO',subclass_counts = subclass_counts,steps=steps)
            modelA.load_state_dict(torch.load('.//models//Best_model_gdro.pth'))
            print("Traditional gDRO trained!")

            over_acc_gdro,gdro1,gdro2,gdro3,gdro4,gdro5 = d_utils.evaluate(test_dataloader,modelB, 5,verbose = True)

            over_acc_gdro_lst.append(over_acc_gdro)
            gdro1_lst.append(gdro1)
            gdro2_lst.append(gdro2)
            gdro3_lst.append(gdro3)
            gdro4_lst.append(gdro4)
            gdro5_lst.append(gdro5)
            
            itemlist =modelA
            with open('./test_results/model_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)
            
            itemlist =test_dataloader
            with open('./test_results/test_dataloader.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist =modelB
            with open('./test_results/model_gdro.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)
            
            itemlist =over_acc_gdro_lst
            with open('./test_results/over_test_acc_gdro.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = gdro1_lst
            with open('./test_results/acc1_gdro.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = gdro2_lst
            with open('./test_results/acc2_gdro.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = gdro3_lst
            with open('./test_results/acc3_gdro.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = gdro4_lst
            with open('./test_results/acc4_gdro.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)
            itemlist = gdro5_lst
            with open('./test_results/acc5_gdro.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = gdro_train
            with open('./test_results/gdro_train_lst.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = gdro_vals
            with open('./test_results/gdro_val_lst.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist =    overall_gdro_vals
            with open('./test_results/gdro_val_overall.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

    elif args.curriculum == 'Five_class':

        for i in range(11): 
            
            i=1
            print("Trial",i)
            params ={'learning_rate': 0.0005,
                                'patience':2,
                                'batch_size': 128,  
                                'w_d': 0.005,
                                'factor': 0.2,
                                'scheduler_choice':1,
                                'opt': 'Adam' }
            
                

            file_num.append(i)
                
            split_file = os.path.join('./data/Train_splits/nodule_split_?.csv').replace("?",str(i))
            
            data_easy,datas_hard = im_utils.get_cur_features(device=DEVICE,file=split_file,mode='unsorted')  

            datas_cur = im_utils.get_erm_features(device=DEVICE,file=split_file,mode='curriculum') 

            _,cv_data,test_data = datas_cur
            

            train_data_easy,cv_data_easy = data_easy
            train_data_hard,cv_data_hard = datas_hard

            trainDataset1 = LIDC_Dataset(*train_data_easy)
            validDataset1 = LIDC_Dataset(*cv_data_easy)

            trainDataset2 = LIDC_Dataset(*train_data_hard)
            validDataset2 = LIDC_Dataset(*cv_data_hard)

            testDataset = LIDC_Dataset(*test_data)

            

            tr = trainDataset1
            val = validDataset1
            test=testDataset

            
            subclass_counts1=trainDataset1.get_class_counts('subclass')

            train_weights1 = im_utils.get_sampler_weights(trainDataset1.subclasses)    


            
            train_dataloader1 = DataLoader(tr, batch_size =params['batch_size'],shuffle=False,sampler=torch.utils.data.WeightedRandomSampler(train_weights1,len(train_weights1)) )
            

            try:
                val_weights1 =   im_utils.get_sampler_weights(validDataset1.subclasses)
            except:
                val_weights1 =   im_utils.get_sampler_weights(validDataset1.labels)

            val_dataloader1 = DataLoader(val,batch_size = len(validDataset1) ,shuffle = False,sampler = torch.utils.data.WeightedRandomSampler(val_weights1,len(val_weights1)) )



            tr = trainDataset2
            val = validDataset2

            subclass_counts2=trainDataset2.get_class_counts('subclass')
            train_weights2 = im_utils.get_sampler_weights(trainDataset2.subclasses)    


            
            train_dataloader2 = DataLoader(tr, batch_size =params['batch_size'],shuffle=False,sampler=torch.utils.data.WeightedRandomSampler(train_weights2,len(train_weights2)) )
            

            val_weights2 =   im_utils.get_sampler_weights(validDataset2.subclasses)
            val_dataloader2 = DataLoader(val,batch_size = len(validDataset2) ,shuffle = False,sampler = torch.utils.data.WeightedRandomSampler(val_weights2,len(val_weights2)) )


            validDataset = LIDC_Dataset(*cv_data)
            testDataset = LIDC_Dataset(*test_data)


            
            val = validDataset
            val_weights =   im_utils.get_sampler_weights(validDataset.subclasses)
            test_weights =   im_utils.get_sampler_weights(testDataset.subclasses)

            
            val_dataloader = DataLoader(val,batch_size = len(validDataset) ,shuffle = False,sampler = torch.utils.data.WeightedRandomSampler(val_weights,len(val_weights)) )
            test_dataloader = DataLoader(test, batch_size = len(testDataset) ,shuffle = False,sampler=torch.utils.data.WeightedRandomSampler(test_weights,len(test_weights)) )  
            
            

            device = torch.device('cuda')

            if args.freeze == 'Yes':
                model = models.TransferModel18(num_classes=2)
                model2 = models.TransferModel18(num_classes=2)  #Initially 2 for matching states and changed later.
            else:
                model = models.TransferModel18(freeze=False,num_classes=2)
                model2 = models.TransferModel18(freeze=False,num_classes=2)
            
            steps1 = math.ceil(len(trainDataset1) / params['batch_size'])
            steps2 = math.ceil(len(trainDataset2) / params['batch_size'])

            modelA,max_acc,cur_train,cur_vals,overall_val = train_gdro_ct_new(params,model,train_dataloader1,val_dataloader1,train_dataloader2,val_dataloader2,num_epochs=50,mode='cur_gDRO',subclass_counts1=subclass_counts1,subclass_counts2=subclass_counts2,steps1=steps1,steps2=steps2)
            modelA.load_state_dict(torch.load('.//models//Best_model_cur_gdro.pth'))
            print("Cur gDRO trained!")
            print(file_num)
            over_acc_cur_gdro,cur_gdro1,cur_gdro2,cur_gdro3,cur_gdro4,cur_gdro5 = d_utils.evaluate(test_dataloader,modelA, 5,verbose = True)
      
            over_acc_cur_gdro_lst.append(over_acc_cur_gdro)
            cur_gdro1_lst.append(cur_gdro1)
            cur_gdro2_lst.append(cur_gdro2)
            cur_gdro3_lst.append(cur_gdro3)
            cur_gdro4_lst.append(cur_gdro4)
            cur_gdro5_lst.append(cur_gdro5)

            itemlist =over_acc_cur_gdro_lst
            with open('./test_results/over_test_acc_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = cur_gdro1_lst
            with open('./test_results/acc1_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = cur_gdro2_lst
            with open('./test_results/acc2_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = cur_gdro3_lst
            with open('./test_results/acc3_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = cur_gdro4_lst
            with open('./test_results/acc4_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = cur_gdro5_lst
            with open('./test_results/acc5_gdro_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)
            
            itemlist = file_num
            with open('./test_results/test_logs.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)
            
            itemlist = cur_train
            with open('./test_results/cur_train_lst.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = cur_vals
            with open('./test_results/cur_val_lst.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            itemlist = overall_val
            with open('./test_results/overall_val_cur.txt', 'wb') as fp:
                pickle.dump(itemlist, fp)

            



        
if method =='ERM':
        
    if args.curriculum =='No':
        over_acc_erm_lst.append(over_acc_erm)
        erm1_lst.append(erm1)
        erm2_lst.append(erm2)
        erm3_lst.append(erm3)
        erm4_lst.append(erm4)
        erm5_lst.append(erm5)
        
        itemlist =over_acc_erm_lst
        with open('./test_results/over_test_acc_erm.txt', 'wb') as fp:
            pickle.dump(itemlist, fp)

        itemlist = erm1_lst
        with open('./test_results/acc1_erm.txt', 'wb') as fp:
            pickle.dump(itemlist, fp)

        itemlist = erm2_lst
        with open('./test_results/acc2_erm.txt', 'wb') as fp:
            pickle.dump(itemlist, fp)

        itemlist = erm3_lst
        with open('./test_results/acc3_erm.txt', 'wb') as fp:
            pickle.dump(itemlist, fp)

        itemlist = erm4_lst
        with open('./test_results/acc4_erm.txt', 'wb') as fp:
            pickle.dump(itemlist, fp)
        itemlist = erm5_lst
        with open('./test_results/acc5_erm.txt', 'wb') as fp:
            pickle.dump(itemlist, fp)

    elif args.curriculum =='Yes':
        
        over_acc_erm_lst_cur.append(over_acc_erm)
        erm1_lst_cur.append(erm1)
        erm2_lst_cur.append(erm2)
        erm3_lst_cur.append(erm3)
        erm4_lst_cur.append(erm4)
        erm5_lst_cur.append(erm5)
        
        itemlist =over_acc_erm_lst_cur
        with open('./test_results/over_test_acc_erm_cur.txt', 'wb') as fp:
            pickle.dump(itemlist, fp)

        itemlist = erm1_lst_cur
        with open('./test_results/acc1_erm_cur.txt', 'wb') as fp:
            pickle.dump(itemlist, fp)

        itemlist = erm2_lst_cur
        with open('./test_results/acc2_erm_cur.txt', 'wb') as fp:
            pickle.dump(itemlist, fp)

        itemlist = erm3_lst_cur
        with open('./test_results/acc3_erm_cur.txt', 'wb') as fp:
            pickle.dump(itemlist, fp)

        itemlist = erm4_lst_cur
        with open('./test_results/acc4_erm_cur.txt', 'wb') as fp:
            pickle.dump(itemlist, fp)

        itemlist = erm5_lst_cur
        with open('./test_results/acc5_erm_cur.txt', 'wb') as fp:
            pickle.dump(itemlist, fp)

        
else:
    pass

if args.significance =='Yes':

        if method =='ERM':

            with open ('./test_results/over_test_acc_erm.txt', 'rb') as fp:
                acc_erm = pickle.load(fp)

            with open ('./test_results/acc1_erm.txt', 'rb') as fp:
                acc_erm1 = pickle.load(fp)

            with open ('./test_results/acc2_erm.txt', 'rb') as fp:
                acc_erm2 = pickle.load(fp)

            with open ('./test_results/acc3_erm.txt', 'rb') as fp:
                acc_erm3 = pickle.load(fp)
    
            with open ('./test_results/acc4_erm.txt', 'rb') as fp:
                acc_erm4 = pickle.load(fp)
    
            with open ('./test_results/acc5_erm.txt', 'rb') as fp:
                acc_erm5 = pickle.load(fp)
    


       
    
            with open ('./test_results/over_test_acc_erm_cur.txt', 'rb') as fp:
                acc_erm_cur = pickle.load(fp)

            with open ('./test_results/acc1_erm_cur.txt', 'rb') as fp:
                acc_erm_cur1 = pickle.load(fp)

            with open ('./test_results/acc2_erm_cur.txt', 'rb') as fp:
                acc_erm_cur2 = pickle.load(fp)

            with open ('./test_results/acc3_erm_cur.txt', 'rb') as fp:
                acc_erm_cur3 = pickle.load(fp)
    
            with open ('./test_results/acc4_erm_cur.txt', 'rb') as fp:
                acc_erm_cur4 = pickle.load(fp)
    
            with open ('./test_results/acc5_erm_cur.txt', 'rb') as fp:
                acc_erm_cur5 = pickle.load(fp)
    


            print("overall accuracy ERM", d_utils.Average(acc_erm), 'trials',len(acc_erm))
            print(d_utils.Average(acc_erm1),d_utils.Average(acc_erm2),d_utils.Average(acc_erm3),d_utils.Average(acc_erm4),d_utils.Average(acc_erm5))

            print("overall accuracy curriculum ERM", d_utils.Average(acc_erm_cur),'trials',len(acc_erm_cur))
            print(d_utils.Average(acc_erm_cur1),d_utils.Average(acc_erm_cur2),d_utils.Average(acc_erm_cur3),d_utils.Average(acc_erm_cur4),d_utils.Average(acc_erm_cur5))

            res = stats.ttest_rel(acc_erm_cur,acc_erm)
    
            display(res)

        elif method == 'gDRO':

            with open ('./test_results/over_test_acc_gdro.txt', 'rb') as fp:
                acc_gdro = pickle.load(fp)

            with open ('./test_results/acc1_gdro.txt', 'rb') as fp:
                acc_gdro1 = pickle.load(fp)

            with open ('./test_results/acc2_gdro.txt', 'rb') as fp:
                acc_gdro2 = pickle.load(fp)

            with open ('./test_results/acc3_gdro.txt', 'rb') as fp:
                acc_gdro3 = pickle.load(fp)
    
            with open ('./test_results/acc4_gdro.txt', 'rb') as fp:
                acc_gdro4 = pickle.load(fp)
    
            with open ('./test_results/acc5_gdro.txt', 'rb') as fp:
                acc_gdro5 = pickle.load(fp)
    


       
    
            with open ('./test_results/over_test_acc_gdro_cur.txt', 'rb') as fp:
                acc_gdro_cur = pickle.load(fp)

            with open ('./test_results/acc1_gdro_cur.txt', 'rb') as fp:
                acc_gdro_cur1 = pickle.load(fp)

            with open ('./test_results/acc2_gdro_cur.txt', 'rb') as fp:
                acc_gdro_cur2 = pickle.load(fp)

            with open ('./test_results/acc3_gdro_cur.txt', 'rb') as fp:
                acc_gdro_cur3 = pickle.load(fp)
    
            with open ('./test_results/acc4_gdro_cur.txt', 'rb') as fp:
                acc_gdro_cur4 = pickle.load(fp)
    
            with open ('./test_results/acc5_gdro_cur.txt', 'rb') as fp:
                acc_gdro_cur5 = pickle.load(fp)
    


            print("overall accuracy gdro", d_utils.Average(acc_gdro), 'trials',len(acc_gdro))
            print(d_utils.Average(acc_gdro1),d_utils.Average(acc_gdro2),d_utils.Average(acc_gdro3),d_utils.Average(acc_gdro4),d_utils.Average(acc_gdro5))

            print("overall accuracy curriculum gdro", d_utils.Average(acc_gdro_cur),'trials',len(acc_gdro_cur))
            print(d_utils.Average(acc_gdro_cur1),d_utils.Average(acc_gdro_cur2),d_utils.Average(acc_gdro_cur3),d_utils.Average(acc_gdro_cur4),d_utils.Average(acc_gdro_cur5))


            print("Overall significance",stats.ttest_rel(acc_gdro_cur,acc_gdro))
            print("Worst group significance",stats.ttest_rel(acc_gdro_cur2,acc_gdro2))
	        

        else:
            pass
else:
    pass