#!/usr/bin/env python
# coding: utf-8

# filesystem and os
import os
import psutil
# gputil
import GPUtil

# pandas/numpy
import pandas as pd
import numpy as np

# data transforms
import umap

# other sklearn
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

# models
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

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
        print(f"Training model {i+1} of {n_warmup} (warmup) for {model_name}.")
        model = model_class(**params)
        model.fit(X, y)
        del model
    
    for i in range(n):
        print(f"Training model {i+1} of {n} for {model_name}.")
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
    
    #print(t_list)
    
    return bm

# config

print("Imports and function definitions done!")

has_gpu = len(GPUtil.getGPUs())>0
num_cpu = os.cpu_count()
gb_ram = psutil.virtual_memory()[0]/2**30

print("Found GPU:", has_gpu)

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


# run benchmark

n_data_list = [400_000, 200_000, 100_000, 50_000]
#n_data_list = [50_000]
result = pd.DataFrame()

for n_data in n_data_list:
    X, y = generateData(n_samples=n_data)
    
    print("Data generated, dimensions of X data:", X.shape)
    if(has_gpu):
        bm = doBenchmark(XGBClassifier, "xgb_gpu", X, y, xgbgpu_paras, n_warmup=1)
        result = result.append(bm, ignore_index=True)

    bm = doBenchmark(XGBClassifier, "xgb_cpu", X, y, xgbcpu_paras, n_warmup=1)
    result = result.append(bm, ignore_index=True)

    bm = doBenchmark(RandomForestClassifier, "rf_cpu", X, y, rf_paras, n_warmup=0)
    result = result.append(bm, ignore_index=True)

    bm = doBenchmark(LogisticRegression, "log_reg_cpu", X, y, {'max_iter':1000}, n_warmup=0)
    result = result.append(bm, ignore_index=True)
    
    if(n_data<=100_000):
        bm = doBenchmark(umap.UMAP, "umap_fit_cpu", X, y, umap_paras, n_warmup=1)
        result = result.append(bm, ignore_index=True)
    
    bm = doBenchmark(PCA, "pca_fit_cpu", X, y, n_warmup=0)
    result = result.append(bm, ignore_index=True)
    
    print("All models done for this dataset!")
    
result['num_cpu'] = num_cpu
result['gb_ram'] = gb_ram
result.to_csv("benchmark.csv", index=False)
