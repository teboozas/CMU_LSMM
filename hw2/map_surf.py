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


with open('/home/ubuntu/11775-hws/hw2_code/surf_features/trn_surf.pkl', 'rb') as f:
    obj = pickle.load(f)
    trn_data_surf = obj[0]
    trn_label_surf = obj[1]
    trn_size_surf = obj[2]
    trn_names_surf = obj[3]
    f.close()

with open('/home/ubuntu/11775-hws/hw2_code/surf_features/val_surf.pkl', 'rb') as f:
    obj = pickle.load(f)
    val_data_surf = obj[0]
    val_label_surf = obj[1]
    val_size_surf = obj[2]
    val_names_surf = obj[3]
    f.close()

with open('/home/ubuntu/11775-hws/hw2_code/surf_features/test_surf.pkl', 'rb') as f:
    obj = pickle.load(f)
    test_data_surf = obj[0]
    test_label_surf = obj[1]
    test_size_surf = obj[2]
    test_names_surf = obj[3]
    f.close()

def get_averaged(data, label, size):
    data_avg_ = []
    label_avg_ = []
    for i in range(len(size)):
        if i == 0:
            tmp_data = data[:size[0]]
            tmp_label = label[:size[0]][0]
        elif i > 0:
            tmp_data = data[np.sum(size[:i]):np.sum(size[:i]) + size[i]]
            tmp_label = label[np.sum(size[:i]):np.sum(size[:i]) + size[i]][0]

        tmp_data = np.average(tmp_data.toarray(), axis = 0)
        data_avg_.append(tmp_data)
        label_avg_.append(tmp_label)
        
    data_avg = np.array(data_avg_)
    label_avg = np.array(label_avg_)
    
    return data_avg, label_avg

trn_data_surf_avg, trn_label_surf_avg = get_averaged(trn_data_surf, trn_label_surf, trn_size_surf)
val_data_surf_avg, val_label_surf_avg = get_averaged(val_data_surf, val_label_surf, val_size_surf)

trn_label_surf_avg_001 = trn_label_surf_avg.copy()
trn_label_surf_avg_001[trn_label_surf_avg != 'P001'] = 'NULL'
trn_label_surf_avg_002 = trn_label_surf_avg.copy()
trn_label_surf_avg_002[trn_label_surf_avg != 'P002'] = 'NULL'
trn_label_surf_avg_003 = trn_label_surf_avg.copy()
trn_label_surf_avg_003[trn_label_surf_avg != 'P003'] = 'NULL'
val_label_surf_avg_001 = val_label_surf_avg.copy()
val_label_surf_avg_001[val_label_surf_avg != 'P001'] = 'NULL'
val_label_surf_avg_002 = val_label_surf_avg.copy()
val_label_surf_avg_002[val_label_surf_avg != 'P002'] = 'NULL'
val_label_surf_avg_003 = val_label_surf_avg.copy()
val_label_surf_avg_003[val_label_surf_avg != 'P003'] = 'NULL'

lgbm_surf_avg_001 = LGBMClassifier(random_state = 0, n_jobs = -1, class_weight = 'balanced',
                               boosting_type = 'goss', n_estimators = 500, learning_rate = 0.002,
                               reg_alpha = 0.5, reg_beta = 0.5, max_depth = 30)

lgbm_surf_avg_001.fit(trn_data_surf_avg, trn_label_surf_avg_001)
val_prob_surf_avg_001 = lgbm_surf_avg_001.predict_proba(val_data_surf_avg).T
map_surf_001 = average_precision_score(y_true = val_label_surf_avg_001, y_score = val_prob_surf_avg_001[1], pos_label = 'P001')

lgbm_surf_avg_002 = LGBMClassifier(random_state = 0, n_jobs = -1, class_weight = 'balanced',
                               boosting_type = 'goss', n_estimators = 500, learning_rate = 0.002,
                               reg_alpha = 0.5, reg_beta = 0.5, max_depth = 30)

lgbm_surf_avg_002.fit(trn_data_surf_avg, trn_label_surf_avg_002)
val_prob_surf_avg_002 = lgbm_surf_avg_002.predict_proba(val_data_surf_avg).T
map_surf_002 = average_precision_score(y_true = val_label_surf_avg_002, y_score = val_prob_surf_avg_002[1], pos_label = 'P002')

lgbm_surf_avg_003 = LGBMClassifier(random_state = 0, n_jobs = -1, class_weight = 'balanced',
                               boosting_type = 'goss', n_estimators = 500, learning_rate = 0.002,
                               reg_alpha = 0.5, reg_beta = 0.5, max_depth = 30)

lgbm_surf_avg_003.fit(trn_data_surf_avg, trn_label_surf_avg_003)
val_prob_surf_avg_003 = lgbm_surf_avg_003.predict_proba(val_data_surf_avg).T
map_surf_003 = average_precision_score(y_true = val_label_surf_avg_003, y_score = val_prob_surf_avg_003[1], pos_label = 'P003')

print("MAP with SURF Features on Validation Set(P001): {} %".format(round(map_surf_001*100, 2)))
print("MAP with SURF Features on Validation Set(P002): {} %".format(round(map_surf_002*100, 2)))
print("MAP with SURF Features on Validation Set(P003): {} %".format(round(map_surf_003*100, 2)))

P001_surf_ = lgbm_surf_avg_001.predict_proba(test_data_surf_avg).T[1]
P002_surf_ = lgbm_surf_avg_002.predict_proba(test_data_surf_avg).T[1]
P003_surf_ = lgbm_surf_avg_003.predict_proba(test_data_surf_avg).T[1]

P001_surf = get_dict(P001_surf_, test_names_surf)
P002_surf = get_dict(P002_surf_, test_names_surf)
P003_surf = get_dict(P003_surf_, test_names_surf)

test_names_surf_sorted = np.sort(test_names_surf.copy())

files = ['P001_surf.lst', 'P002_surf.lst', 'P003_surf.lst']
pairs = [P001_surf, P002_surf, P003_surf]
names = [test_names_surf_sorted, test_names_surf_sorted, test_names_surf_sorted]

try:
    os.mkdir('scores')
except:
    pass

for idx in range(len(names)):
    with open(os.path.join('scores', files[idx]), 'w') as f:
        for name in names[idx]:
            f.write("{}\n".format(pairs[idx][name]))
        f.close()