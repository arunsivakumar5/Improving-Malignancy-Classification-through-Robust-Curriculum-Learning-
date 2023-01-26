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
from IPython.display import clear_output


def get_sampler_weights(subclass_labels):
    '''
    Returns a list of weights that allows uniform sampling of dataset
    by subclasses
    '''
    
    subclasses = torch.unique(subclass_labels)
    subclass_freqs = []

    for subclass in subclasses:
        subclass_counts = sum(subclass_labels == subclass)
        subclass_freqs.append(1 / subclass_counts)

    subclass_weights = torch.zeros_like(subclass_labels).float()
    

    for idx, label in enumerate(subclass_labels):
            if idx ==0:
                pass
            else:
                subclass_weights[idx] = subclass_freqs[int(label)]

    return subclass_weights

def get_normed(this_array, this_min=0, this_max=255, set_to_int=True):
    """
        INPUTS:
        this_array: raw image from file
        OUTPUT:
        normalized version of image
    """

    rat = (this_max - this_min) / (this_array.max() - this_array.min())
    this_array = this_array * rat
    this_array -= this_array.min()
    this_array += this_min
    if set_to_int:
        return this_array.to(dtype=torch.int) / this_max
    return this_array / this_max


def scale_image(image_dim, upscale_amount=None, crop_change=None):
    """
        INPUTS:
        upscale_amount: amount to upscale image by, if None, upscales
                 to original size
        OUTPUTS:
        scalar: a function that returns the multichannel scaled version
                of a image
    """
    if not upscale_amount:
        upscale_amount = image_dim

    if not crop_change:
        crop_change = image_dim // 4

    crop_1_amount = image_dim
    crop_2_amount = image_dim - crop_change
    crop_3_amount = image_dim - 2 * crop_change

    upscale = torchvision.transforms.Resize(upscale_amount)
    crop_1 = torchvision.transforms.CenterCrop(crop_1_amount)
    crop_2 = torchvision.transforms.CenterCrop(crop_2_amount)
    crop_3 = torchvision.transforms.CenterCrop(crop_3_amount)

    def scalar(image):
        """
            INPUTS:
            Image: normalized image of shape (1, H, W)
            NOTE: H should equal W
            OUPUTS:
            scaled image: image with channels of different crops of
                          image, shape of (3, H, W)
        """

        img_ch1 = upscale(crop_1(image))
        img_ch2 = upscale(crop_2(image))
        img_ch3 = upscale(crop_3(image))
        image = torch.cat([img_ch1, img_ch2, img_ch3])

        return image

    return scalar


def get_malignancy(lidc_df, nodule_id, binary, device='cuda'):

    malignancy = lidc_df[lidc_df['noduleID'] == nodule_id]['malignancy'].iloc[0]
 
    if binary:
        return torch.tensor(1, device=device) if malignancy > 1 else torch.tensor(0)

    return torch.tensor(malignancy) if malignancy > 1 else torch.tensor(malignancy)


def get_subclass(lidc_df, nodule_id, sublabels):
    subtype = lidc_df[lidc_df['noduleID'] == nodule_id][sublabels].iloc[0]
    return torch.tensor(subtype)



def get_data_split(train_test_df, nodule_id):
    return torch.tensor(train_test_df[train_test_df['noduleID'] == nodule_id]['split'].iloc[0])


def augment_image(image):
    """
        Input:
        image: tensor of shape (3, H, W)
        Ouput:
        tuple of image and its augmented versions
    """

    image_90 = torchvision.transforms.functional.rotate(image, 90)
    image_180 = torchvision.transforms.functional.rotate(image, 180)
    image_270 = torchvision.transforms.functional.rotate(image, 270)
    image_f = torch.flip(image, [0, 1])  # flip along x-axis

    return image, image_90, image_180, image_270, image_f


def get_images(root, paths, transform=transforms.ToTensor(), device='cpu'):
            img_tensors = []
            for img_path in tqdm(paths):
                img = Image.open(root + img_path)
                img_tensors.append(transform(img).to(device))
                img.close()
            return torch.stack(img_tensors)

def images_to_df(image_folder='./data/LIDC(MaxSlices)_Nodules',
                 image_labels='./data/LIDC_20130817_AllFeatures2D_MaxSlicePerNodule_inLineRatings.csv',
                 image_dim=71):
                 
                 
                 
    
    '''
    Creates a dataframe of noduleID and corresponding image tensor
    inputs:
    image_folder: the folder of images
    image_labels: path of csv file containing labels
    image_dim: the dimension of image
    output:
    img_df: pandas dataframe of noduleID and corresponding image as a numpy tensor
    '''
    
    
    
    Cur_labels = pd.read_csv('./data/LIDC_3_4_Ratings_wMSE.csv', index_col=0)
    LIDC_labels = pd.read_csv(image_labels, index_col=0)
    LIDC_labels = LIDC_labels[LIDC_labels['noduleID'].isin(Cur_labels ['noduleID'])]
    
    LIDC_labels.sort_values('noduleID', inplace=True)
    Cur_labels.sort_values('noduleID', inplace=True)
    Cur_labels.reset_index(drop=True, inplace=True)
    LIDC_labels.reset_index(drop=True, inplace=True)
    
    
    LIDC_labels['malignancy'] = Cur_labels['Malignancy']
    LIDC_labels['curriculum'] = Cur_labels['curriculum']
    LIDC_labels['wMSE'] = Cur_labels['wMSE']
    
    
    
    scalar = scale_image(image_dim)
    cols = {'noduleID': [], 'malignancy': [], 'image': [], 'curriculum': [],'wMSE':[]}

    for file in os.listdir(image_folder):
        nodule_id = int(file.split('.')[0]) 
          
        malignancy =LIDC_labels.loc[LIDC_labels['noduleID'] == nodule_id, 'malignancy'].values
        cur =LIDC_labels.loc[LIDC_labels['noduleID'] == nodule_id, 'curriculum'].values
        wmse =LIDC_labels.loc[LIDC_labels['noduleID'] == nodule_id, 'wMSE'].values
        
        
        
        image_raw = np.loadtxt(os.path.join(image_folder, file))
        image_raw = torch.from_numpy(image_raw)
        image_normed = get_normed(image_raw).unsqueeze(dim=0)
        image = scalar(image_normed)

        cols['noduleID'].append(nodule_id)
        cols['malignancy'].append(malignancy.tolist())
        cols['curriculum'].append(cur.tolist())
        cols['wMSE'].append(wmse.tolist())
        cols['image'].append(image)
        

    img_df = pd.DataFrame(cols)
    img_df.sort_values('noduleID', inplace=True)
    img_df['malignancy'] = img_df['malignancy'].astype(str).apply(lambda x: x.replace('[','').replace(']',''))
    img_df['curriculum'] = img_df['curriculum'].astype(str).apply(lambda x: x.replace('[','').replace(']',''))
    img_df['wMSE'] = img_df['wMSE'].astype(str).apply(lambda x: x.replace('[','').replace(']',''))
    img_df.reset_index(drop=True, inplace=True)
    img_df.replace('', np.nan, inplace=True)
    img_df.dropna(inplace=True)
    img_df.reset_index(inplace=True,drop=True)
    


    return img_df


def sort_df(images_df,mode='default'):

    
    
    if mode=='default':
        images_df.sort_values(by=['wMSE'],inplace=True)
        images_df.reset_index(drop=True,inplace=True)

        
    else:
        images_df1 = images_df[images_df['malignancy']== 0 ]
        images_df2 = images_df[images_df['malignancy']== 1 ]
        images_df3 = images_df[images_df['malignancy']== 2 ]
        images_df4 = images_df[images_df['malignancy']== 3 ]
        images_df5 = images_df[images_df['malignancy']== 4 ]
    
        images_df1.sort_values(by=['wMSE'],inplace=True)
        images_df1.reset_index(drop=True,inplace=True)

        images_df2.sort_values(by=['wMSE'],inplace=True)
        images_df2.reset_index(drop=True,inplace=True)

        images_df3.sort_values(by=['wMSE'],inplace=True)
        images_df3.reset_index(drop=True,inplace=True)

        images_df4.sort_values(by=['wMSE'],inplace=True)
        images_df4.reset_index(drop=True,inplace=True)

        images_df5.sort_values(by=['wMSE'],inplace=True)
        images_df5.reset_index(drop=True,inplace=True)
    
        df_new = pd.DataFrame()
        for i in range(len(images_df)):
            idx =i 
            df_new = df_new.append([images_df1[images_df1.index==idx],images_df2[images_df2.index==idx],images_df3[images_df3.index==idx],images_df4[images_df4.index==idx],images_df5[images_df5.index==idx]])
        df_new.reset_index(inplace=True,drop=True)
        images_df= df_new
        
    return images_df


def get_erm_features(file='./data/LIDC_3_4_Ratings_wMSE.csv',
                 subclass_file=None,
                 images=False,
                 features=None,
                 device='cuda',
                 subclass='cluster',mode='traditional'):

    '''
    gets features in their train, cv and test splits
    inputs:
    feature_file: path of csv file with features
    split_file: path of csv with data split of nodules
    subclass_file: path of csv with the subclasses of nodules
    images: if True get images instead of using feature_file
    features: only checked if images == True, if not None, should be the df of
              noduleID and corresponding image tensors
    device: cpu or cuda, device to place the tensors on
    sublcass: the name of column in subclass_file to extract subclass labels from
    outputs:
    datas: a list of size three, of train data, cross val data, test data
           datas[i]: a list of size three, of features, labels, subclass labels
    '''

    if mode =='split':

        df_splits = pd.read_csv(file, index_col=0)
        mask_var = df_splits['wMSE'] == 0

        

        df_splits1 = df_splits[mask_var]
        df_splits1.reset_index(drop=True, inplace=True)
        df_splits2 = df_splits[~mask_var]
        df_splits2.reset_index(drop=True, inplace=True)

        df_features = images_to_df()
    

        df_features = df_features.loc[df_features["curriculum"] == "0"]
        df_features.reset_index(inplace=True,drop = True)
        
    
        df_features['clusters'] = df_features['malignancy']
    
        df_features.sort_values('noduleID', inplace=True)
        df_features.reset_index(drop=True, inplace=True)
    

        def label_cls (row):
                if row['clusters'] < 3 :
                    return 0
                else:
                    return 1
        def label_cls_subclass (row):
                if row['clusters']==1 :
                    return 0
                elif row['clusters']==2:
                    return 1
                elif row['clusters']==3:
                    return 2
                elif row['clusters']==4:
                    return 3
                elif row['clusters']==5:
                    return 4
    
    
        df_features['cur_cls'] = df_features['clusters'].astype(int)
    
        df_features['clusters'] = df_features['clusters'].astype(int)
        df_features['malignancy_b'] =  df_features.apply(lambda row: label_cls(row), axis=1)
        df_features['malignancy'] =  df_features.apply(lambda row: label_cls_subclass(row), axis=1)

        df_features['clusters'] = df_features['malignancy']

    
    
    
        
        df_splits1 = df_splits1[df_splits1['noduleID'].isin(df_features['noduleID'])]

        df_splits1.sort_values('noduleID', inplace=True)
        df_splits1.reset_index(drop=True, inplace=True)
    
        dfs = []
    
        for i in range(3):
            dfs.append(df_features.loc[(df_splits1['splits'] == i).values])

        datas = []

        # If we choose to do curriculum learning, sort data based on wMSE computed from multiple radiologist labels
        dfs2=[]
    
        for i in dfs:
            i = sort_df(i)
            dfs2.append(i)
        
        print(dfs2)
        for i, d in enumerate(dfs2):
                # If the training dataset, we need to do data augmentation
                if i == 0:
                    
                    imgs = []
                    for img in d['image']:
                        imgs.extend(augment_image(img))
                    X = torch.stack(imgs).to(device=device, dtype=torch.float32)

                    # hacky way to repeat the labels for the additional augmented images
                    augments = X.shape[0] // len(d)
                    d_temp = pd.DataFrame()
                    d_temp['cur_cls'] = np.repeat(d['cur_cls'].values, augments)
                    d_temp['malignancy_b'] = np.repeat(d['malignancy_b'].values, augments)
                    d_temp['clusters'] = np.repeat(d['clusters'].values, augments)
                    d = d_temp
                    d.reset_index(drop=True, inplace=True)
                else:
                    X = torch.stack(list(d['image'])).to(device=device, dtype=torch.float32)
                

                y = torch.tensor(d['malignancy_b'].values, device=device, dtype=torch.long)
                c = torch.tensor(d['clusters'].values, device=device, dtype=torch.long)
            
               
                datas.append((X, y, c))

        

        print( "easy",d['malignancy_b'].value_counts())
        df_features = images_to_df()
    

    
        
    
        df_features['clusters'] = df_features['malignancy']
    
        df_features.sort_values('noduleID', inplace=True)
        df_features.reset_index(drop=True, inplace=True)
    

        def label_cls (row):
                if row['clusters'] < 3 :
                    return 0
                elif row['clusters'] == 3 :
                    return 1
                else:
                    return 2
        def label_cls_subclass (row):
                if row['clusters']==1 :
                    return 0
                elif row['clusters']==2:
                    return 1
                elif row['clusters']==3:
                    return 2
                elif row['clusters']==4:
                    return 3
                elif row['clusters']==5:
                    return 4
    
    
        df_features['cur_cls'] = df_features['clusters'].astype(int)
    
        df_features['clusters'] = df_features['clusters'].astype(int)
        df_features['malignancy_b'] =  df_features.apply(lambda row: label_cls(row), axis=1)
        df_features['malignancy'] =  df_features.apply(lambda row: label_cls_subclass(row), axis=1)

        df_features['clusters'] = df_features['malignancy']

        
    
    
        df_splits1 = pd.read_csv(file,index_col=0)
        df_splits1 = df_splits1[df_splits1['noduleID'].isin(df_features['noduleID'])]

        df_splits1.sort_values('noduleID', inplace=True)
        df_splits1.reset_index(drop=True, inplace=True)
    
        dfs = []
    
        for i in range(3):
            dfs.append(df_features.loc[(df_splits1['splits'] == i).values])

        datas2 = []

        # If we choose to do curriculum learning, sort data based on wMSE computed from multiple radiologist labels
        dfs2=[]
    
        for i in dfs:
            i = sort_df(i)
            dfs2.append(i)
        
        print(dfs2)
        for i, d in enumerate(dfs2):
                # If the training dataset, we need to do data augmentation
                if i == 0:
                    
                    imgs = []
                    for img in d['image']:
                        imgs.extend(augment_image(img))
                    X = torch.stack(imgs).to(device=device, dtype=torch.float32)

                    # hacky way to repeat the labels for the additional augmented images
                    augments = X.shape[0] // len(d)
                    d_temp = pd.DataFrame()
                    d_temp['cur_cls'] = np.repeat(d['cur_cls'].values, augments)
                    d_temp['malignancy_b'] = np.repeat(d['malignancy_b'].values, augments)
                    d_temp['clusters'] = np.repeat(d['clusters'].values, augments)
                    d = d_temp
                    d.reset_index(drop=True, inplace=True)
                else:
                    X = torch.stack(list(d['image'])).to(device=device, dtype=torch.float32)
                

                y = torch.tensor(d['malignancy_b'].values, device=device, dtype=torch.long)
                c = torch.tensor(d['clusters'].values, device=device, dtype=torch.long)
            
               
                datas2.append((X, y, c))

        return datas,datas2
            
    else:
        df_features = images_to_df()
    

    
        df_splits = pd.read_csv(file, index_col=0)
        df_splits.reset_index(inplace=True,drop = True)
    
        df_features['clusters'] = df_features['malignancy']
    
        df_features.sort_values('noduleID', inplace=True)
        df_features.reset_index(drop=True, inplace=True)
    

        def label_cls (row):
                if row['clusters'] < 3 :
                    return 0
                elif row['clusters'] == 3 :
                    return 1
                else:
                    return 2
        def label_cls_subclass (row):
                if row['clusters']==1 :
                    return 0
                elif row['clusters']==2:
                    return 1
                elif row['clusters']==3:
                    return 2
                elif row['clusters']==4:
                    return 3
                elif row['clusters']==5:
                    return 4
    
    
        df_features['cur_cls'] = df_features['clusters'].astype(int)
    
        df_features['clusters'] = df_features['clusters'].astype(int)
        df_features['malignancy_b'] =  df_features.apply(lambda row: label_cls(row), axis=1)
        df_features['malignancy'] =  df_features.apply(lambda row: label_cls_subclass(row), axis=1)

        df_features['clusters'] = df_features['malignancy']

    
    
    
        df_splits = pd.read_csv(file,index_col=0)
        df_splits = df_splits[df_splits['noduleID'].isin(df_features['noduleID'])]

        df_splits.sort_values('noduleID', inplace=True)
        df_splits.reset_index(drop=True, inplace=True)
    
        dfs = []
    
        for i in range(3):
            dfs.append(df_features.loc[(df_splits['splits'] == i).values])

        datas = []

        # If we choose to do curriculum learning, sort data based on wMSE computed from multiple radiologist labels
        if mode=='curriculum':
            dfs2=[]
    
            for i in dfs:
                i = sort_df(i)
                dfs2.append(i)
        
            print(dfs2)
            for i, d in enumerate(dfs2):
                    # If the training dataset, we need to do data augmentation
                    if i == 0:
                    
                        imgs = []
                        for img in d['image']:
                            imgs.extend(augment_image(img))
                        X = torch.stack(imgs).to(device=device, dtype=torch.float32)

                        # hacky way to repeat the labels for the additional augmented images
                        augments = X.shape[0] // len(d)
                        d_temp = pd.DataFrame()
                        d_temp['cur_cls'] = np.repeat(d['cur_cls'].values, augments)
                        d_temp['malignancy_b'] = np.repeat(d['malignancy_b'].values, augments)
                        d_temp['clusters'] = np.repeat(d['clusters'].values, augments)
                        d = d_temp
                        d.reset_index(drop=True, inplace=True)
                    else:
                        X = torch.stack(list(d['image'])).to(device=device, dtype=torch.float32)
                

                    y = torch.tensor(d['malignancy_b'].values, device=device, dtype=torch.long)
                    c = torch.tensor(d['clusters'].values, device=device, dtype=torch.long)
            
               
                    datas.append((X, y, c))
        else:
            for i, d in enumerate(dfs):
                    if i == 0:
                    
                        imgs = []
                        for img in d['image']:
                            imgs.extend(augment_image(img))
                        X = torch.stack(imgs).to(device=device, dtype=torch.float32)

                        # hacky way to repeat the labels for the additional augmented images
                        augments = X.shape[0] // len(d)
                        d_temp = pd.DataFrame()
                        d_temp['cur_cls'] = np.repeat(d['cur_cls'].values, augments)
                        d_temp['malignancy_b'] = np.repeat(d['malignancy_b'].values, augments)
                        d_temp['clusters'] = np.repeat(d['clusters'].values, augments)
                        d = d_temp
                        d.reset_index(drop=True, inplace=True)
                    else:
                        X = torch.stack(list(d['image'])).to(device=device, dtype=torch.float32)
                

                    y = torch.tensor(d['malignancy_b'].values, device=device, dtype=torch.long)
                    c = torch.tensor(d['clusters'].values, device=device, dtype=torch.long)
            
               
                    datas.append((X, y, c))
    
        return datas
def random_split(prop,split_file):
    
    prop = prop/100
    
    
    
    df_splits = split_file

    df_splits = df_splits[df_splits['splits'] <= 1]
    
    
    df1 = df_splits[df_splits['Malignancy'] == 1 ]
    df2 = df_splits[df_splits['Malignancy'] == 2 ]
    df3 = df_splits[df_splits['Malignancy'] == 3 ]
    df4 = df_splits[df_splits['Malignancy'] == 4 ]
    df5 = df_splits[df_splits['Malignancy'] == 5 ]

    df_sample = df1.sample(frac=1)
    print(df_splits)
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
    df_split1.reset_index(drop=True, inplace=True)
    
    df_split2.sort_values('noduleID', inplace=True)
    df_split2.reset_index(drop=True, inplace=True)
    
    #df_split1 = (df_split1.iloc[: , 1:]).reset_index(drop = True)
    #df_split2 = (df_split2.iloc[: , 1:]).reset_index(drop = True)

    
    return df_split1,df_split2
def get_cur_features(file='./data/LIDC_3_4_Ratings_wMSE.csv',
                 subclass_file=None,
                 images=False,
                 features=None,
                 device='cuda',
                 subclass='cluster',mode='traditional'):

    '''
    gets features in their train, cv and test splits
    inputs:
    feature_file: path of csv file with features
    split_file: path of csv with data split of nodules
    subclass_file: path of csv with the subclasses of nodules
    images: if True get images instead of using feature_file
    features: only checked if images == True, if not None, should be the df of
              noduleID and corresponding image tensors
    device: cpu or cuda, device to place the tensors on
    sublcass: the name of column in subclass_file to extract subclass labels from
    outputs:
    datas: a list of size three, of train data, cross val data, test data
           datas[i]: a list of size three, of features, labels, subclass labels
    '''

    if mode =='experiment1_unsorted':

        df_splits = pd.read_csv(file, index_col=0)

        

        split1,split2 = random_split(50,split_file=df_splits)

        df_splits1 = split1
        df_splits2 = split2

        df_features = images_to_df()
    

        #df_features = df_features.loc[df_features["curriculum"] == "0"]
        df_features.reset_index(inplace=True,drop = True)
        
    
        df_features['clusters'] = df_features['malignancy']
    
        df_features.sort_values('noduleID', inplace=True)
        df_features.reset_index(drop=True, inplace=True)
    

        def label_cls (row):
                if row['clusters'] == 3 :
                    return 1
                else:
                    return 0
        def label_cls_subclass (row):
                if row['clusters']==1 :
                    return 0
                elif row['clusters']==2:
                    return 1
                elif row['clusters']==3:
                    return 2
                elif row['clusters']==4:
                    return 3
                elif row['clusters']==5:
                    return 4
    
    
        df_features['cur_cls'] = df_features['clusters'].astype(int)
    
        df_features['clusters'] = df_features['clusters'].astype(int)
        df_features['malignancy_b'] =  df_features.apply(lambda row: label_cls(row), axis=1)
        df_features['malignancy'] =  df_features.apply(lambda row: label_cls_subclass(row), axis=1)

        df_features['clusters'] = df_features['malignancy']

    
    
        print(df_splits1)
        
        df_splits1 = df_splits1[df_splits1['noduleID'].isin(df_features['noduleID'])]
        df_splits1.sort_values('noduleID', inplace=True)
        df_splits1.reset_index(drop=True, inplace=True)
    
        df_features = df_features[df_features['noduleID'].isin(df_splits1['noduleID'])]
        df_features.sort_values('noduleID', inplace=True)
        df_features.reset_index(drop=True, inplace=True)

        dfs = []
    
        for i in range(3):
            dfs.append(df_features.loc[(df_splits1['splits'] == i).values])

        datas = []

        # If we choose to do curriculum learning, sort data based on wMSE computed from multiple radiologist labels
        #dfs2=[]
    
        #for i in dfs:
            #i = sort_df(i)
            #dfs2.append(i)
        
        #print(dfs2)
        for i, d in enumerate(dfs):
                # If the training dataset, we need to do data augmentation
                if i == 0:
                    
                    imgs = []
                    for img in d['image']:
                        imgs.extend(augment_image(img))
                    X = torch.stack(imgs).to(device=device, dtype=torch.float32)

                    # hacky way to repeat the labels for the additional augmented images
                    augments = X.shape[0] // len(d)
                    d_temp = pd.DataFrame()
                    d_temp['cur_cls'] = np.repeat(d['cur_cls'].values, augments)
                    d_temp['malignancy_b'] = np.repeat(d['malignancy_b'].values, augments)
                    d_temp['clusters'] = np.repeat(d['clusters'].values, augments)
                    d = d_temp
                    d.reset_index(drop=True, inplace=True)
                else:
                    X = torch.stack(list(d['image'])).to(device=device, dtype=torch.float32)
                

                y = torch.tensor(d['malignancy_b'].values, device=device, dtype=torch.long)
                c = torch.tensor(d['clusters'].values, device=device, dtype=torch.long)
            
               
                datas.append((X, y, c))

        

        print( "easy",d['malignancy_b'].value_counts())
        df_features = images_to_df()
    

    
        
    
        df_features['clusters'] = df_features['malignancy']
    
        df_features.sort_values('noduleID', inplace=True)
        df_features.reset_index(drop=True, inplace=True)
    

        def label_cls (row):
                if row['clusters'] < 3 :
                    return 0
                elif row['clusters'] == 3 :
                    return 1
                else:
                    return 2
        def label_cls_subclass (row):
                if row['clusters']==1 :
                    return 0
                elif row['clusters']==2:
                    return 1
                elif row['clusters']==3:
                    return 2
                elif row['clusters']==4:
                    return 3
                elif row['clusters']==5:
                    return 4
    
    
        df_features['cur_cls'] = df_features['clusters'].astype(int)
    
        df_features['clusters'] = df_features['clusters'].astype(int)
        df_features['malignancy_b'] =  df_features.apply(lambda row: label_cls(row), axis=1)
        df_features['malignancy'] =  df_features.apply(lambda row: label_cls_subclass(row), axis=1)

        df_features['clusters'] = df_features['malignancy']

        
    
        df_splits1 = df_splits2
        df_splits1 = pd.read_csv(file,index_col=0)

        df_splits1 = df_splits1[df_splits1['noduleID'].isin(df_features['noduleID'])]
        df_splits1.sort_values('noduleID', inplace=True)
        df_splits1.reset_index(drop=True, inplace=True)


        df_features = df_features[df_features['noduleID'].isin(df_splits1['noduleID'])]
        df_features.sort_values('noduleID', inplace=True)
        df_features.reset_index(drop=True, inplace=True)

        
    
        dfs = []
    
        for i in range(3):
            dfs.append(df_features.loc[(df_splits1['splits'] == i).values])

        datas2 = []

        # If we choose to do curriculum learning, sort data based on wMSE computed from multiple radiologist labels
        #dfs2=[]
    
        #for i in dfs:
            #i = sort_df(i)
            #dfs2.append(i)
        
        #print(dfs2)
        for i, d in enumerate(dfs):
                # If the training dataset, we need to do data augmentation
                if i == 0:
                    
                    imgs = []
                    for img in d['image']:
                        imgs.extend(augment_image(img))
                    X = torch.stack(imgs).to(device=device, dtype=torch.float32)

                    # hacky way to repeat the labels for the additional augmented images
                    augments = X.shape[0] // len(d)
                    d_temp = pd.DataFrame()
                    d_temp['cur_cls'] = np.repeat(d['cur_cls'].values, augments)
                    d_temp['malignancy_b'] = np.repeat(d['malignancy_b'].values, augments)
                    d_temp['clusters'] = np.repeat(d['clusters'].values, augments)
                    d = d_temp
                    d.reset_index(drop=True, inplace=True)
                else:
                    X = torch.stack(list(d['image'])).to(device=device, dtype=torch.float32)
                

                y = torch.tensor(d['malignancy_b'].values, device=device, dtype=torch.long)
                c = torch.tensor(d['clusters'].values, device=device, dtype=torch.long)
            
               
                datas2.append((X, y, c))

        return datas,datas2
            
    else:
        df_features = images_to_df()
    

    
        df_splits = pd.read_csv(file, index_col=0)
        df_splits.reset_index(inplace=True,drop = True)
    
        df_features['clusters'] = df_features['malignancy']
    
        df_features.sort_values('noduleID', inplace=True)
        df_features.reset_index(drop=True, inplace=True)
    

        def label_cls (row):
                if row['clusters'] < 3 :
                    return 0
                elif row['clusters'] == 3 :
                    return 1
                else:
                    return 2
        def label_cls_subclass (row):
                if row['clusters']==1 :
                    return 0
                elif row['clusters']==2:
                    return 1
                elif row['clusters']==3:
                    return 2
                elif row['clusters']==4:
                    return 3
                elif row['clusters']==5:
                    return 4
    
    
        df_features['cur_cls'] = df_features['clusters'].astype(int)
    
        df_features['clusters'] = df_features['clusters'].astype(int)
        df_features['malignancy_b'] =  df_features.apply(lambda row: label_cls(row), axis=1)
        df_features['malignancy'] =  df_features.apply(lambda row: label_cls_subclass(row), axis=1)

        df_features['clusters'] = df_features['malignancy']

    
    
    
        df_splits = pd.read_csv(file,index_col=0)
        df_splits = df_splits[df_splits['noduleID'].isin(df_features['noduleID'])]

        df_splits.sort_values('noduleID', inplace=True)
        df_splits.reset_index(drop=True, inplace=True)
    
        dfs = []
    
        for i in range(3):
            dfs.append(df_features.loc[(df_splits['splits'] == i).values])

        datas = []

        # If we choose to do curriculum learning, sort data based on wMSE computed from multiple radiologist labels
        if mode=='curriculum':
            dfs2=[]
    
            for i in dfs:
                i = sort_df(i)
                dfs2.append(i)
        
            print(dfs2)
            for i, d in enumerate(dfs2):
                    # If the training dataset, we need to do data augmentation
                    if i == 0:
                    
                        imgs = []
                        for img in d['image']:
                            imgs.extend(augment_image(img))
                        X = torch.stack(imgs).to(device=device, dtype=torch.float32)

                        # hacky way to repeat the labels for the additional augmented images
                        augments = X.shape[0] // len(d)
                        d_temp = pd.DataFrame()
                        d_temp['cur_cls'] = np.repeat(d['cur_cls'].values, augments)
                        d_temp['malignancy_b'] = np.repeat(d['malignancy_b'].values, augments)
                        d_temp['clusters'] = np.repeat(d['clusters'].values, augments)
                        d = d_temp
                        d.reset_index(drop=True, inplace=True)
                    else:
                        X = torch.stack(list(d['image'])).to(device=device, dtype=torch.float32)
                

                    y = torch.tensor(d['malignancy_b'].values, device=device, dtype=torch.long)
                    c = torch.tensor(d['clusters'].values, device=device, dtype=torch.long)
            
               
                    datas.append((X, y, c))
        else:
            for i, d in enumerate(dfs):
                    if i == 0:
                    
                        imgs = []
                        for img in d['image']:
                            imgs.extend(augment_image(img))
                        X = torch.stack(imgs).to(device=device, dtype=torch.float32)

                        # hacky way to repeat the labels for the additional augmented images
                        augments = X.shape[0] // len(d)
                        d_temp = pd.DataFrame()
                        d_temp['cur_cls'] = np.repeat(d['cur_cls'].values, augments)
                        d_temp['malignancy_b'] = np.repeat(d['malignancy_b'].values, augments)
                        d_temp['clusters'] = np.repeat(d['clusters'].values, augments)
                        d = d_temp
                        d.reset_index(drop=True, inplace=True)
                    else:
                        X = torch.stack(list(d['image'])).to(device=device, dtype=torch.float32)
                

                    y = torch.tensor(d['malignancy_b'].values, device=device, dtype=torch.long)
                    c = torch.tensor(d['clusters'].values, device=device, dtype=torch.long)
            
               
                    datas.append((X, y, c))
    
        return datas