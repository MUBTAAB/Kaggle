# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:54:13 2017

@author: mucs_b
"""

import numpy as np
import pandas as pd
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn import grid_search

print('import')
train = pd.read_csv('C:/Users/mucs_b/Desktop/Python projects/Local zillow/train_2016.csv')
prop = pd.read_csv('C:/Users/mucs_b/Desktop/Python projects/Local zillow/properties_2016.csv')
sample = pd.read_csv('C:/Users/mucs_b/Desktop/Python projects/Local zillow/sample_submission.csv')

print('merging train and prop')
for i, dtype in zip(prop.columns, prop.dtypes):
	if dtype == np.float64:
		prop[i] = prop[i].astype(np.float32)
df_train = train.merge(prop, how='left', on='parcelid')

print('date')
df_train["transactiondate"] = pd.to_datetime(df_train["transactiondate"])
df_train["year"] = df_train["transactiondate"].dt.year
df_train["month"] = df_train["transactiondate"].dt.month
df_train["day"] = df_train["transactiondate"].dt.day

print('fillna')
df_train=df_train.fillna(0)

print('XY_train')
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode','fireplaceflag','hashottuborspa'], axis=1)
y_train = df_train['logerror'].values

print('labelencoder')
for f in x_train.columns: 
    if x_train[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(list(x_train[f].values)) 
        x_train[f] = lbl.transform(list(x_train[f].values))
 
split = 80000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]   
    
from sklearn import ensemble
model = ensemble.ExtraTreesRegressor(n_estimators=5, max_depth=10, max_features=0.3, n_jobs=-1, random_state=0)
model.fit(x_train, y_train)

from sklearn import model_selection
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = ensemble.ExtraTreesRegressor(n_estimators=5, max_depth=10, max_features=0.3, n_jobs=-1, random_state=0)
results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold)

