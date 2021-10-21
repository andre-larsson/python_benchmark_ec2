#!/usr/bin/env python
# coding: utf-8

# pandas/numpy
import pandas as pd
import numpy as np

# file management and os
import os
from pathlib import Path
from pprint import pprint
import glob
import shutil

# data transforms
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import umap

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# other sklearn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.validation import check_array, check_X_y
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

# models
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# other
from time import perf_counter as p_f


# define functions
def generateData(n_samples=100_000, n_features=60, noise_scale=10):

    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=int(n_features*0.5), n_redundant=int(n_features*0.2))
    
    # add noise with random means/std to features
    rng = np.random.default_rng(111)
    means = rng.uniform(-noise_scale/2,noise_scale/2, n_features)*X.mean(axis=0)
    stds = rng.uniform(0,noise_scale, n_features)*X.std(axis=0)
    X = X+ rng.normal(means, stds, X.shape)
    return X, y

def doBenchmark(model_class, model_name, X, y, params={}, n=5, n_warmup=1):
    
    t_list = []
    
    # for e.g. xgboost gpu, first round always takes longer
    for i in range(n_warmup):
        model = model_class(**params)
        model.fit(X, y)
        del model
    
    for i in range(n):
        print(i)
        model = model_class(**params)
        t0=p_f()
        model.fit(X, y)
        t_list += [p_f()-t0]
        del model
        
    bm = {'model': model_name,
          'n_obs': X.shape[0],
          'n_features': X.shape[1],
        't_mean': np.mean(t_list),
        't_std': np.std(t_list)
         }
    
    print(t_list)
    
    return bm


# define parameter sets

rf_paras = {'n_jobs':-1,
            'verbose':1,
            'n_estimators' : 100}

xgbcpu_paras = {'learning_rate': 0.01,
                'subsample': 0.7,
                'n_estimators': 100,# 3931
                'max_depth': 5,
                'verbosity': 1,
                'colsample_bytree': 0.5,
                'use_label_encoder' : False,
                'tree_method' : 'hist',
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'n_jobs':-1
               }
xgbgpu_paras = {'learning_rate': 0.01,
                'subsample': 0.7,
                'n_estimators': 100,# 3931
                'max_depth': 5,
                'verbosity': 1,
                'colsample_bytree': 0.5,
                'use_label_encoder' : False,
                'tree_method' : 'gpu_hist',
                'objective': 'binary:logistic',
                'eval_metric': 'auc'
               }

umap_paras = {
    'random_state' : 111
}


# In[4]:


get_ipython().run_cell_magic('time', '', '\nn_data_list = [50_000, 100_000, 200_000]\nresult = pd.DataFrame()\n\nfor n_data in n_data_list:\n    X, y = generateData(n_samples=n_data)\n\n    bm = doBenchmark(XGBClassifier, "xgb_gpu", X, y, xgbgpu_paras, n_warmup=1)\n    result = result.append(bm, ignore_index=True)\n\n    bm = doBenchmark(XGBClassifier, "xgb_cpu", X, y, xgbcpu_paras, n_warmup=1)\n    result = result.append(bm, ignore_index=True)\n\n    bm = doBenchmark(RandomForestClassifier, "rf_cpu", X, y, rf_paras, n_warmup=0)\n    result = result.append(bm, ignore_index=True)\n\n    bm = doBenchmark(LogisticRegression, "log_reg_cpu", X, y, {\'max_iter\':1000}, n_warmup=0)\n    result = result.append(bm, ignore_index=True)\n    \n    if(n_data<=100_000):\n        bm = doBenchmark(umap.UMAP, "umap_fit_cpu", X, y, umap_paras, n_warmup=0)\n        result = result.append(bm, ignore_index=True)\n    \n    bm = doBenchmark(PCA, "pca_fit_cpu", X, y, n_warmup=0)\n    result = result.append(bm, ignore_index=True)\n    \n    \n    bm = doBenchmark(StandardScaler, "standard_scaler_fit_cpu", X, y, n_warmup=1)\n    result = result.append(bm, ignore_index=True)\n    \n\nresult.to_csv("benchmark.csv", index=False)\nresult')

