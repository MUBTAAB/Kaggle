# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:52:20 2017

@author: mucs_b
"""

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn import grid_search

print('Loading data ...')

train = pd.read_csv('C:/Users/mucs_b/Desktop/Python projects/Local zillow/train_2016.csv')
prop = pd.read_csv('C:/Users/mucs_b/Desktop/Python projects/Local zillow/properties_2016.csv')

for c, dtype in zip(prop.columns, prop.dtypes):	
    if dtype == np.float64:		
        prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train

"""
for i in x_train.columns:
    print(i + str(sum(pd.isnull(x_train[i]))/len(x_train[i])))
"""

for c in x_train.columns:
    x_train[str(c)+'isnan'] = pd.isnull(x_train[c])
    #x_train[c] = x_train[c].fillna(x_train[c].mode()[0])
    x_train[c] = x_train[c].fillna(0)

param_grid = {
                 'n_estimators': [1000],
                 'max_depth': [2,10,1000],
                 'min_samples_leaf': [1000]
             }

clf = RandomForestRegressor()
grid_clf = grid_search.GridSearchCV(clf, param_grid, cv=10, verbose = 5)
grid_clf.fit(x_train, y_train)

print('best params:' + str(grid_clf.best_params_))
print('best params:' + str(grid_clf.best_score_))

print("Prepare for the prediction ...")
sample = pd.read_csv('C:/Users/mucs_b/Desktop/Python projects/Local zillow/sample_submission.csv')
sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')
del sample;

x_test = df_test[train_columns]

for c in x_test.columns:
    x_test[str(c)+'isnan'] = pd.isnull(x_test[c])
    #x_test[c] = x_test[c].fillna(x_test[c].mode()[0])
    x_test[c] = x_test[c].fillna(0)

del df_test; 

for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
x_test = x_test.values.astype(np.float32, copy=False)


print("Start prediction ...")
p_test = grid_clf.predict(x_test)
#p_test = 0.93*p_test + 0.065*0.012


del x_test

print("Start write result ...")
sub = pd.read_csv('C:/Users/mucs_b/Desktop/Python projects/Local zillow/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

sub.to_csv('rf_starter.csv', index=False, float_format='%.4f')

print("Job 's done ...")


    
    



