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

with open('/home/ubuntu/11775-hws/hw2_code/cnn_features/trn_cnn.pkl', 'rb') as f:
    obj = pickle.load(f)
    trn_data_cnn = obj[0]
    trn_label_cnn = obj[1]
    trn_size_cnn = obj[2]
    trn_names_cnn = obj[3]
    f.close()

with open('/home/ubuntu/11775-hws/hw2_code/cnn_features/val_cnn.pkl', 'rb') as f:
    obj = pickle.load(f)
    val_data_cnn = obj[0]
    val_label_cnn = obj[1]
    val_size_cnn = obj[2]
    val_names_cnn = obj[3]
    f.close()

with open('/home/ubuntu/11775-hws/hw2_code/cnn_features/test_cnn.pkl', 'rb') as f:
    obj = pickle.load(f)
    test_data_cnn = obj[0]
    test_label_cnn = obj[1]
    test_size_cnn = obj[2]
    test_names_cnn = obj[3]
    f.close()

trn_label_cnn_001 = trn_label_cnn.copy()
trn_label_cnn_001[trn_label_cnn != 'P001'] = 'NULL'
trn_label_cnn_002 = trn_label_cnn.copy()
trn_label_cnn_002[trn_label_cnn != 'P002'] = 'NULL'
trn_label_cnn_003 = trn_label_cnn.copy()
trn_label_cnn_003[trn_label_cnn != 'P003'] = 'NULL'
val_label_cnn_001 = val_label_cnn.copy()
val_label_cnn_001[val_label_cnn != 'P001'] = 'NULL'
val_label_cnn_002 = val_label_cnn.copy()
val_label_cnn_002[val_label_cnn != 'P002'] = 'NULL'
val_label_cnn_003 = val_label_cnn.copy()
val_label_cnn_003[val_label_cnn != 'P003'] = 'NULL'

lgbm_cnn_001 = LGBMClassifier(random_state = 0, n_jobs = -1, class_weight = 'balanced',
                               boosting_type = 'goss', n_estimators = 500, learning_rate = 0.002,
                               reg_alpha = 0.5, reg_beta = 0.5, max_depth = 30)

lgbm_cnn_001.fit(trn_data_cnn, trn_label_cnn_001)
val_prob_cnn_001 = lgbm_cnn_001.predict_proba(val_data_cnn).T
map_cnn_001 = average_precision_score(y_true = val_label_cnn_001, y_score = val_prob_cnn_001[1], pos_label = 'P001')

lgbm_cnn_002 = LGBMClassifier(random_state = 0, n_jobs = -1, class_weight = 'balanced',
                               boosting_type = 'goss', n_estimators = 500, learning_rate = 0.002,
                               reg_alpha = 0.5, reg_beta = 0.5, max_depth = 30)

lgbm_cnn_002.fit(trn_data_cnn, trn_label_cnn_002)
val_prob_cnn_002 = lgbm_cnn_002.predict_proba(val_data_cnn).T
map_cnn_002 = average_precision_score(y_true = val_label_cnn_002, y_score = val_prob_cnn_002[1], pos_label = 'P002')

lgbm_cnn_003 = LGBMClassifier(random_state = 0, n_jobs = -1, class_weight = 'balanced',
                               boosting_type = 'goss', n_estimators = 500, learning_rate = 0.002,
                               reg_alpha = 0.5, reg_beta = 0.5, max_depth = 30)

lgbm_cnn_003.fit(trn_data_cnn, trn_label_cnn_003)
val_prob_cnn_003 = lgbm_cnn_003.predict_proba(val_data_cnn).T
map_cnn_003 = average_precision_score(y_true = val_label_cnn_003, y_score = val_prob_cnn_003[1], pos_label = 'P003')

print("MAP with CNN Features on Validation Set(P001): {} %".format(round(map_cnn_001*100, 2)))
print("MAP with CNN Features on Validation Set(P002): {} %".format(round(map_cnn_002*100, 2)))
print("MAP with CNN Features on Validation Set(P003): {} %".format(round(map_cnn_003*100, 2)))

P001_cnn_ = lgbm_cnn_001.predict_proba(test_data_cnn).T[1]
P002_cnn_ = lgbm_cnn_002.predict_proba(test_data_cnn).T[1]
P003_cnn_ = lgbm_cnn_003.predict_proba(test_data_cnn).T[1]

def get_dict(prob, names):
    tmp = {}
    for i in range(len(prob)):
        tmp[names[i]] = prob[i]
    return tmp

P001_cnn = get_dict(P001_cnn_, test_names_cnn)
P002_cnn = get_dict(P002_cnn_, test_names_cnn)
P003_cnn = get_dict(P003_cnn_, test_names_cnn)

P001_cnn['HVC5781'] = 0.0
P002_cnn['HVC5781'] = 0.0
P003_cnn['HVC5781'] = 0.0
test_names_cnn.append('HVC5781')

test_names_cnn_sorted = np.sort(test_names_cnn.copy())

files = ['P001_cnn.lst', 'P002_cnn.lst', 'P003_cnn.lst']
pairs = [P001_cnn, P002_cnn, P003_cnn]
names = [test_names_cnn_sorted, test_names_cnn_sorted, test_names_cnn_sorted]

try:
    os.mkdir('scores')
except:
    pass

for idx in range(len(names)):
    with open(os.path.join('scores', files[idx]), 'w') as f:
        for name in names[idx]:
            f.write("{}\n".format(pairs[idx][name]))
        f.close()