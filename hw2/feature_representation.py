import os
import sys
import threading
import cv2
import numpy as np
import yaml
import pickle
import pdb

import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from pathlib import Path
from scipy.sparse import csr_matrix
from collections import Counter
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import chi2_kernel, laplacian_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


train_path = '/home/ubuntu/11775-hws/all_trn.lst'
val_path = '/home/ubuntu/11775-hws/all_val.lst'
test_path = '/home/ubuntu/11775-hws/all_test_fake.lst'

surf_path = '/home/ubuntu/11775-hws/hw2_code/surf'
cnn_path = '/home/ubuntu/11775-hws/hw2_code/cnn'

def get_splits(split_path):
    split = {}
    with open(split_path, 'r') as f:
        for line in f:
            line = line.split('\n')[0]
            split[line.split(' ')[0]] = line.split(' ')[1]
    return split

def get_split_features(split, type_path):
    
    filenames = list(split.keys())
    i = 0
    size = []
    names = []
    for filename_ in os.listdir(type_path):
        
        filename = filename_.split('.')[0]
        if filename in filenames:
            
            i += 1
            if i % 100 == 0:
                print("Processing: {}th file".format(i))
            try:
                with open(os.path.join(type_path, filename_), 'rb') as f:
                    obj = pickle.load(f)[0][0]
                f.close()
                
                if (type(obj) == type(None)) and ('surf' in type_path):
                    data_ = np.array([0.0] * 64).reshape(1,-1)
                    label_ = np.array(['NULL'])
                elif (type(obj) == type(None)) and ('cnn' in type_path):
                    data_ = np.array([0.0] * 1000).reshape(1,-1)
                    label_ = np.array(['NULL'])
                elif (type(obj) != type(None)) and ('surf' in type_path):
                    data_ = obj
                    label_ = np.array([split[filename]] * obj.shape[0])
                elif (type(obj) != type(None)) and ('cnn' in type_path):
                    data_ = obj.reshape(1,-1)
                    label_ = np.array([split[filename]])
                    
                    
            except:
                if 'surf' in type_path:
                    data_ = np.array([0.0] * 64).reshape(1,-1)
                elif 'cnn' in type_path:
                    data_ = np.array([0.0] * 1000).reshape(1,-1)
                label_ = np.array(['NULL'])
                
            if i == 1:
                data = data_
                label = label_
            elif i > 1:
                data = np.concatenate((data, data_), axis = 0)
                label = np.concatenate((label, label_))
            
            size.append(data_.shape[0])
            names.append(filename)
                
    data = csr_matrix(data)

    return data, label, size, names

trn = get_splits(train_path)
val = get_splits(val_path)
test = get_splits(test_path)

trn_data_surf, trn_label_surf, trn_size_surf, trn_names_surf = get_split_features(trn, surf_path)
val_data_surf, val_label_surf, val_size_surf, val_names_surf = get_split_features(val, surf_path)
test_data_surf, test_label_surf, test_size_surf, test_names_surf = get_split_features(test, surf_path)
trn_data_cnn, trn_label_cnn, trn_size_cnn, trn_names_cnn = get_split_features(trn, cnn_path)
val_data_cnn, val_label_cnn, val_size_cnn, val_names_cnn = get_split_features(val, cnn_path)
test_data_cnn, test_label_cnn, test_size_cnn, test_names_cnn = get_split_features(test, cnn_path)

os.mkdir('/home/ubuntu/11775-hws/hw2_code/surf_features')
os.mkdir('/home/ubuntu/11775-hws/hw2_code/cnn_features')

trn_surf = [trn_data_surf, trn_label_surf, trn_size_surf, trn_names_surf]
with open('/home/ubuntu/11775-hws/hw2_code/surf_features/trn_surf.pkl', 'wb') as f:
    pickle.dump(trn_surf, f)

val_surf = [val_data_surf, val_label_surf, val_size_surf, val_names_surf]
with open('/home/ubuntu/11775-hws/hw2_code/surf_features/val_surf.pkl', 'wb') as f:
    pickle.dump(val_surf, f)

test_surf = [test_data_surf, test_label_surf, test_size_surf, test_names_surf]
with open('/home/ubuntu/11775-hws/hw2_code/surf_features/test_surf.pkl', 'wb') as f:
    pickle.dump(test_surf, f)

trn_cnn = [trn_data_cnn, trn_label_cnn, trn_size_cnn, trn_names_cnn]
with open('/home/ubuntu/11775-hws/hw2_code/cnn_features/trn_cnn.pkl', 'wb') as f:
    pickle.dump(trn_cnn, f)

val_cnn = [val_data_cnn, val_label_cnn, val_size_cnn, val_names_cnn]
with open('/home/ubuntu/11775-hws/hw2_code/cnn_features/val_cnn.pkl', 'wb') as f:
    pickle.dump(val_cnn, f)

test_cnn = [test_data_cnn, test_label_cnn, test_size_cnn, test_names_cnn]
with open('/home/ubuntu/11775-hws/hw2_code/cnn_features/test_cnn.pkl', 'wb') as f:
    pickle.dump(test_cnn, f)

