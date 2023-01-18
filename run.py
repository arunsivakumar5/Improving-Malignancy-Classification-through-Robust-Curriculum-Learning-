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
from train import train_erm,train_gdro





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


acc_list_new1 = []

all_spec_ls = []
all_sens_ls = []


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
parser.add_argument('--prop', '-t', default=100, help='The proprtion by which the dataset is to be split. The given proprtion goes to classifier retraining Note: This parameter is needed for CRIS ')
parser.add_argument('--curriculum', action='store',  help='If curriculum information has to be used to sort the instances by Easy to hard as data is fed into the classiifer ')
parser.add_argument('--significance', action='store',  help='If significance tests should be carried out between a set of classifier ')


args = parser.parse_args()


method = args.method
if method =='random':
    prop = int(args.prop)
else:
    pass






DEVICE =  torch.device('cuda')





if method =='ERM':

    params ={
        'learning_rate': 0.01,
        'patience':90,
        'batch_size': 64,
        'w_d': 0.3,
        'factor': 0.3,
        'scheduler_choice': 2,
        'opt': 'SGD' }
        
    params2 ={
        'learning_rate': 0.1,
        'patience': 20,
        'batch_size': 1024,
        'w_d': 0.9,
        'factor': 0.3,
        'scheduler_choice': 1,
        'opt': 'SGD'  }
        
    if args.curriculum == 'Both':

        for i in range(1,args.trials + 1): 
            
            split_file = os.path.join('./data/Train_splits/nodule_split_?.csv').replace("?",str(i))
            
            datas_cur = im_utils.get_erm_features(device=DEVICE,file=split_file,mode='curriculum')

            train_data,cv_data,test_data = datas_cur

            trainDataset = LIDC_Dataset(*train_data)
            validDataset = LIDC_Dataset(*cv_data)
            testDataset = LIDC_Dataset(*test_data)


            tr = trainDataset
            val = validDataset
            test=testDataset

            sampler = SequentialSampler(trainDataset)

            train_dataloader = DataLoader(tr, batch_size =params2['batch_size'],sampler=sampler )

            val_dataloader = DataLoader(val,batch_size = len(validDataset),shuffle = False, num_workers=0)
            test_dataloader = DataLoader(test, batch_size = len(testDataset) , shuffle = False, num_workers=0)   

            device = torch.device('cuda')

            model = models.TransferModel18()

            modelA,max_acc = train_erm(params2,train_dataloader,val_dataloader,model,num_epochs=150,mode='cur_erm')
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
            validDataset = LIDC_Dataset(*cv_data)
            testDataset = LIDC_Dataset(*test_data)


            tr = trainDataset
            val = validDataset
            test= testDataset

            train_weights = im_utils.get_sampler_weights(trainDataset.subclasses)    


            sampler = torch.utils.data.WeightedRandomSampler(
                        train_weights,
                        len(train_weights))
            train_dataloader = DataLoader(tr, batch_size =params['batch_size'],sampler=sampler )
            val_dataloader = DataLoader(val,batch_size = len(validDataset),shuffle = False, num_workers=0)
            test_dataloader = DataLoader(test, batch_size = len(testDataset) , shuffle = False, num_workers=0)

            device = torch.device('cuda')
            model = models.TransferModel18()

            modelA,max_acc = train_erm(params,train_dataloader,val_dataloader,model,num_epochs=150,mode='erm')
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

        
        
            
    if args.curriculum == 'Yes':
        datas = im_utils.get_erm_features(device=DEVICE,file=split_file,mode='curriculum')
    else:
        datas = im_utils.get_erm_features(device=DEVICE,file=split_file,mode='traditional')

        
    train_data,cv_data,test_data = datas

    trainDataset = LIDC_Dataset(*train_data)
    validDataset = LIDC_Dataset(*cv_data)
    testDataset = LIDC_Dataset(*test_data)


    tr = trainDataset
    val = validDataset
    test=testDataset


        
        
    if args.curriculum == 'Yes':

        sampler = SequentialSampler(trainDataset)
        train_dataloader = DataLoader(tr, batch_size =params2['batch_size'],sampler=sampler )

    else:
        train_weights = im_utils.get_sampler_weights(trainDataset.subclasses)    


        sampler = torch.utils.data.WeightedRandomSampler(
                    train_weights,
                    len(train_weights))
        train_dataloader = DataLoader(tr, batch_size =params['batch_size'],sampler=sampler )

    val_dataloader = DataLoader(val,batch_size = len(validDataset),shuffle = False, num_workers=0)
    test_dataloader = DataLoader(test, batch_size = len(testDataset) , shuffle = False, num_workers=0)    
        

    device = torch.device('cuda')




        
    model = models.TransferModel18()




    #Training ERM model
    if args.curriculum == 'Yes':
        modelA,max_acc = train_erm(params2,train_dataloader,val_dataloader,model,num_epochs=150,mode='cur_erm')
        modelA.load_state_dict(torch.load('.//models//Best_model_cur_erm.pth'))
        print("Cur ERM trained!")
    else:
        modelA,max_acc = train_erm(params,train_dataloader,val_dataloader,model,num_epochs=150,mode='erm')
        modelA.load_state_dict(torch.load('.//models//Best_model_erm.pth'))
        print("Traditional ERM trained!")

    over_acc_erm,erm1,erm2,erm3,erm4,erm5 = d_utils.evaluate(test_dataloader,modelA, 5,verbose = True)

elif method =='gdro':

    if args.curriculum == 'Both':

        for i in range(1,args.trials + 1): 

            params ={'learning_rate': 0.0005,
                    'patience':2,
                    'batch_size': 128,
                    'w_d': 0.005,
                    'factor': 0.2,
                    'scheduler_choice':1,
                    'opt': 'Adam' 
                    }
            params2={'learning_rate': 0.0001,
                    'patience':60,
                    'batch_size': 256,
                    'w_d': 0.2,
                    'factor': 0.6,
                    'scheduler_choice':1,
                    'opt': 'SGD' 
                    }
            split_file = os.path.join('./data/Train_splits/nodule_split_?.csv').replace("?",str(i))
            
            datas_cur = im_utils.get_erm_features(device=DEVICE,file=split_file,mode='curriculum')

            train_data,cv_data,test_data = datas_cur

            trainDataset = LIDC_Dataset(*train_data)
            validDataset = LIDC_Dataset(*cv_data)
            testDataset = LIDC_Dataset(*test_data)


            tr = trainDataset
            val = validDataset
            test=testDataset

            sampler = SequentialSampler(trainDataset)

            train_dataloader = DataLoader(tr, batch_size =params2['batch_size'],sampler=sampler )

            val_dataloader = DataLoader(val,batch_size = len(validDataset),shuffle = False, num_workers=0)
            test_dataloader = DataLoader(test, batch_size = len(testDataset) , shuffle = False, num_workers=0)   

            device = torch.device('cuda')

            model = models.TransferModel18()

            modelA,max_acc = train_gdro(params2,model,train_dataloader,val_dataloader,num_epochs=150,mode='cur_gDRO')
            modelA.load_state_dict(torch.load('.//models//Best_model_cur_gdro.pth'))
            print("Cur gDRO trained!")
      
            

            datas = im_utils.get_erm_features(device=DEVICE,file=split_file,mode='traditional')

            train_data,cv_data,test_data = datas

            trainDataset = LIDC_Dataset(*train_data)
            validDataset = LIDC_Dataset(*cv_data)
            testDataset = LIDC_Dataset(*test_data)


            tr = trainDataset
            val = validDataset
            test= testDataset

            train_weights = im_utils.get_sampler_weights(trainDataset.subclasses)    


            sampler = torch.utils.data.WeightedRandomSampler(
                        train_weights,
                        len(train_weights))
            train_dataloader = DataLoader(tr, batch_size =params['batch_size'],sampler=sampler )
            val_dataloader = DataLoader(val,batch_size = len(validDataset),shuffle = False, num_workers=0)
            test_dataloader = DataLoader(test, batch_size = len(testDataset) , shuffle = False, num_workers=0)

            device = torch.device('cuda')
            model = models.TransferModel18()

            modelA,max_acc = train_gdro(params,train_dataloader,val_dataloader,model,num_epochs=150,mode ='gDRO')
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

else:

    meta_train,meta_valid, meta_test,root,transform = utils.get_celeba_ondemand_datasets(device=DEVICE, subclass_label=True)

    df_sample = meta_train.sample(frac=1)

    df_sample0 = df_sample[df_sample['y'] == 0 ]
    df_sample1 = df_sample[df_sample['y'] == 1 ]
        
    df_size = int(0.50*len(df_sample0))
    df_size1 = int(0.50*len(df_sample1))

    metadata_00 = df_sample0[:df_size]
    metadata_01 = df_sample0[df_size:]

    metadata_10 = df_sample1[:df_size1]
    metadata_11 = df_sample1[df_size1:]

    meta_train1  = pd.concat([metadata_00,metadata_10], axis=0,sort = True,ignore_index=True)
    meta_train2 = pd.concat([metadata_01, metadata_11], axis=0,sort = True,ignore_index=True)

    v_sample = meta_valid.sample(frac=1)

    df_sample0 = v_sample[v_sample['y'] == 0 ]
    df_sample1 = v_sample[v_sample['y'] == 1 ]
        
    df_size = int(0.50*len(df_sample0))
    df_size1 = int(0.50*len(df_sample1))

    metadata_00 = df_sample0[:df_size]
    metadata_01 = df_sample0[df_size:]

    metadata_10 = df_sample1[:df_size1]
    metadata_11 = df_sample1[df_size1:]

    meta_valid1  = pd.concat([metadata_00,metadata_10], axis=0,sort = True,ignore_index=True)
    meta_valid2 = pd.concat([metadata_01, metadata_11], axis=0,sort = True,ignore_index=True)


    meta_train1 = meta_train1.reset_index(drop = True)
    meta_valid1 = meta_valid1.reset_index(drop =True)


    meta_train2 = meta_train2.reset_index(drop = True)
    meta_valid2 = meta_valid2.reset_index(drop =True)

    try:
        meta_train1 = meta_train1.drop(columns='index')
        meta_train2 = meta_train2.drop(columns='index')
        meta_valid1 = meta_valid1.drop(columns='index')
        meta_valid2 = meta_valid2.drop(columns='index')

        columnsTitles = ['image_id','partition','y',  'place' 	]
        meta_train1 = meta_train1.reindex(columns=columnsTitles)
        meta_train2 = meta_train2.reindex(columns=columnsTitles)
        meta_valid1 = meta_valid1.reindex(columns=columnsTitles)
        meta_valid2 = meta_valid2.reindex(columns=columnsTitles)
        
        print("after", meta_train1)
    except:
        print("no indexing needed", meta_train1)

    train_dataset = SubclassedDataset_getinstances(meta_train1,root,transform,mode ='train', subclass_label=False)
    val_dataset = SubclassedDataset2(meta_valid1,root,transform,mode = 'val', subclass_label=False)
    test_dataset = SubclassedDataset2(meta_test,root,transform,mode ='test', subclass_label=False)



    tr = train_dataset
    val = val_dataset
    test=test_dataset

    train_dataloader = InfiniteDataLoader(tr, 32,weights=utils.get_sampler_weights(tr.labels))
    val_dataloader = InfiniteDataLoader(val,512,weights=utils.get_sampler_weights(val.labels))
    test_dataloader = InfiniteDataLoader(test, 512,weights=utils.get_sampler_weights(test.subclasses))    
        

    device = torch.device('cuda')




    model_args= {'num_labels':2,'pretrained':True, 'freeze': True, 'device': 'cuda'}
    model = models.TransferModel50(*model_args)




    #Training ERM model
    modelA,avg_loss,val_loss_list,trainids1 = train_feature_ext(train_dataloader,val_dataloader,model,num_epochs=2,mode ='same_instances',trainset = train_dataset)
    modelA.load_state_dict(torch.load('.//models//Best_model_erm.pth'))
    print("ERM trained!")

    over_acc_erm,erm1,erm2,erm3,erm4 = utils.evaluate(test_dataloader,modelA, 4,verbose = True)


    print(trainids1)


    import copy



    model_args= {'num_labels':2,'pretrained':True, 'freeze': True, 'device': 'cuda'}
    modelB = models.TransferModel50(*model_args)




    modelB.load_state_dict(modelA.state_dict())

    #Checking if the weights and bias of the new intialized model match the ERM feature extractor model weights in each layer.
    for (nameA, paramA), (nameB, paramB) in zip(modelA.named_parameters(), modelB.named_parameters()):
        if (paramA == paramB).all():
            print('{} matches {}'.format(nameA, nameB))
        else:
            print('{} does not match {}'.format(nameA, nameB))

    train_dataset2 = SubclassedDataset2(meta_train2,root,transform,mode ='train', subclass_label=False)
    val_dataset2 = SubclassedDataset2(meta_valid2,root,transform,mode = 'val', subclass_label=False)
        



    tr = train_dataset2
    val = val_dataset2

    train_dataloader2 = InfiniteDataLoader(tr, 32,weights=utils.get_sampler_weights(tr.subclasses))
    val_dataloader2 = InfiniteDataLoader(val,512,weights=utils.get_sampler_weights(val.subclasses))
         

    modelB,avg_loss_cris,val_loss_list_cris,tl,trainids2 = train_gdro(train_dataloader2, val_dataloader2,test_dataloader,modelB, num_epochs = 10,mode ='same_instances',trainset = train_dataset)
    print("gDRO trained!")
    modelB.load_state_dict(torch.load('.//models//Best_model_crois.pth'))


        

        

    over_test_acc,acc1,acc2,acc3,acc4 = utils.evaluate(test_dataloader,modelB, 4,verbose = True)
    print("Random CRIS trained!")
    over_acc_cris_rand.append(over_test_acc)
    cris_rand_1.append(acc1)
    cris_rand_2.append(acc2)
    cris_rand_3.append(acc3)
    cris_rand_4.append(acc4)

        
    itemlist =over_acc_cris_rand
    with open('./test_results/over_test_acc_cris.txt', 'wb') as fp:
        pickle.dump(itemlist, fp)




    itemlist = cris_rand_1
    with open('./test_results/acc1_cris.txt', 'wb') as fp:
        pickle.dump(itemlist, fp)




    itemlist = cris_rand_2
    with open('./test_results/acc2_cris.txt', 'wb') as fp:
        pickle.dump(itemlist, fp)




    itemlist = cris_rand_3
    with open('./test_results/acc3_cris.txt', 'wb') as fp:
        pickle.dump(itemlist, fp)

    itemlist = cris_rand_4
    with open('./test_results/acc4_cris.txt', 'wb') as fp:
        pickle.dump(itemlist, fp)

    trainids1.extend(trainids2)
    trainids1 = list(trainids1)
    s = set()
        
    for i in trainids1:
        s.add(i)
        
    trainids =  list(s)
    print("length of chosen random data",len(trainids))
        
    meta_train_ids = meta_train.iloc[trainids].reset_index(drop=True)
    #meta_train_ids = pd.DataFrame(trainids,columns =['image_id'])



    #meta_train = pd.merge(meta_train,meta_train_ids, on='image_id', how="outer")
    meta_train = meta_train.fillna(False)

        

    prop = prop

    meta_train1,meta_valid1,meta_train2,meta_valid2 = utils.get_celeba_central_splits_new(prop= prop,train=meta_train, val = meta_valid,device=DEVICE, subclass_label=True,instances='same',root = root,transform=transform)


    train_dataset = SubclassedDataset2(meta_train1,root,transform,mode ='train', subclass_label=False)
    val_dataset = SubclassedDataset2(meta_valid1,root,transform,mode = 'val', subclass_label=False)
    test_dataset = SubclassedDataset2(meta_test,root,transform,mode ='test', subclass_label=False)



    tr = train_dataset
    val = val_dataset
    test=test_dataset

    train_dataloader = InfiniteDataLoader(tr, 32,weights=utils.get_sampler_weights(tr.labels))
    val_dataloader = InfiniteDataLoader(val,512,weights=utils.get_sampler_weights(val.labels))
    test_dataloader = InfiniteDataLoader(test, 512,weights=utils.get_sampler_weights(test.subclasses))    
        

    device = torch.device('cuda')




    model_args= {'num_labels':2,'pretrained':True, 'freeze': True, 'device': 'cuda'}
    model = models.TransferModel50(*model_args)




    #Training ERM model
    modelA,avg_loss,val_loss_list = train_feature_ext(train_dataloader,val_dataloader,model,num_epochs=2)
    modelA.load_state_dict(torch.load('.//models//Best_model_erm.pth'))
    print("ERM trained!")

    over_acc_erm,erm1,erm2,erm3,erm4 = utils.evaluate(test_dataloader,modelA, 4,verbose = True)





    import copy



    model_args= {'num_labels':2,'pretrained':True, 'freeze': True, 'device': 'cuda'}
    modelB = models.TransferModel50(*model_args)




    modelB.load_state_dict(modelA.state_dict())

    #Checking if the weights and bias of the new intialized model match the ERM feature extractor model weights in each layer.
    for (nameA, paramA), (nameB, paramB) in zip(modelA.named_parameters(), modelB.named_parameters()):
        if (paramA == paramB).all():
            print('{} matches {}'.format(nameA, nameB))
        else:
            print('{} does not match {}'.format(nameA, nameB))

    train_dataset = SubclassedDataset2(meta_train2,root,transform,mode ='train', subclass_label=False)
    val_dataset = SubclassedDataset2(meta_valid2,root,transform,mode = 'val', subclass_label=False)
        



    tr = train_dataset
    val = val_dataset

    train_dataloader2 = InfiniteDataLoader(tr, 32,weights=utils.get_sampler_weights(tr.subclasses))
    val_dataloader2 = InfiniteDataLoader(val,512,weights=utils.get_sampler_weights(val.subclasses))


        
    modelB,avg_loss_cris,val_loss_list_cris,tl,trainids2 = train_gdro(train_dataloader2, val_dataloader2,test_dataloader,modelB, num_epochs = 2)
    print("gDRO trained!")
    modelB.load_state_dict(torch.load('.//models//Best_model_crois.pth'))



       

        

    over_test_acc,acc1,acc2,acc3,acc4 = utils.evaluate(test_dataloader,modelB, 4,verbose = True)

    over_acc_cris_rep.append(over_test_acc)
    cris_rep_1.append(acc1)
    cris_rep_2.append(acc2)
    cris_rep_3.append(acc3)
    cris_rep_4.append(acc4)


    itemlist = over_acc_cris_rep
    with open('./test_results/over_test_acc_cris_rep.txt', 'wb') as fp:
        pickle.dump(itemlist, fp)




    itemlist = cris_rep_1
    with open('./test_results/acc1_cris_rep.txt', 'wb') as fp:
        pickle.dump(itemlist, fp)




    itemlist = cris_rep_2
    with open('./test_results/acc2_cris_rep.txt', 'wb') as fp:
        pickle.dump(itemlist, fp)




    itemlist = cris_rep_3
    with open('./test_results/acc3_cris_rep.txt', 'wb') as fp:
        pickle.dump(itemlist, fp)






    itemlist = cris_rep_4

    with open('./test_results/acc4_cris.txt', 'wb') as fp:
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

        e   
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
	
	        print("Overall significance")
	        res = stats.ttest_rel(acc_gdro_cur,acc_gdro)
            print(res)
	
	        print("Worst group significance")
	        res = stats.ttest_rel(acc_gdro_cur2,acc_gdro2)
            print(res)

        else:
            pass
else:
    pass