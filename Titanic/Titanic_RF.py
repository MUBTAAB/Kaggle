# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:54:12 2017

@author: mucs_b
"""
#Dataprep source: https://www.kaggle.com/sinakhorami/titanic-best-working-classifier
#import modules & data
import pandas as pd 
import numpy as np
import re

source = 'C:/Users/mucs_b/Desktop/Python pet projects/Kotelezo_Titanic_Korok/'
df = pd.read_csv(source+'train.csv', sep = ',')
df_2 = pd.read_csv(source+'test.csv', sep = ',')
full_data = [df, df_2]

#Feature engineering
#Teljes családméret
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

#Egyedül utazik-e?
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1


for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    

for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())

#életkor kitöltése random számmal átlag +- 1 szórás között
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)


def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex & Embarked
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    

#Feature selection 
print(df.columns)
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Fare', 'Embarked']]
y = df['Survived']

from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.33, random_state = 0)

#Model no.1.
from sklearn.tree import DecisionTreeClassifier
estimator1 = DecisionTreeClassifier()
estimator1.fit(Xtrain,ytrain) 
yPred = estimator1.predict(Xtest)
print(np.mean(yPred == ytest))


#n_estimators = 10, max_depth = 10**6, min_samples_leaf = 10
from sklearn.ensemble import RandomForestClassifier
estimator2 = RandomForestClassifier()
estimator2.fit(Xtrain,ytrain) 
yPred = estimator2.predict(Xtest)
print(np.mean(yPred == ytest))

#Model no.3.
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
cv = cross_validation.KFold(len(y), n_folds=10)
estimator3 = RandomForestClassifier(n_estimators = 10)

errors = []
for traincv, testcv in cv:
        estimator3.fit(X.loc[traincv], y[traincv])
        yPred = estimator3.predict(X.loc[testcv])
        errors.append(np.mean(yPred == y.loc[testcv]))

print(errors)        
print(np.mean(errors))
yPred = estimator3.predict(Xtest)
print(np.mean(yPred == ytest))

#Model no.4.
#best params:{'max_depth': 5, 'min_impurity_split': 0, 'min_samples_leaf': 10, 'n_estimators': 100}
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import grid_search

param_grid = {'n_estimators': [10,100,1000],
                 'max_depth': [2,4,8,10],
                 'min_samples_leaf': [1,10,100],
                 'min_impurity_split':[0,0.1,0.2,0.5]}

clf = RandomForestClassifier()
grid_clf = grid_search.GridSearchCV(clf, param_grid, cv=10, verbose = 0)
grid_clf.fit(X, y)

print('best params:' + str(grid_clf.best_params_))
print('best params:' + str(grid_clf.best_score_))

yPred = grid_clf.predict(Xtest)
print(np.mean(yPred == ytest))

source = 'C:/Users/mucs_b/Desktop/Python pet projects/Kotelezo_Titanic_Korok/'

X_subm = df_2[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Fare', 'Embarked']]

Subm1 = pd.DataFrame({'PassengerId':df_2['PassengerId'],
                      'Survived':estimator1.predict(X_subm)})

Subm2 = pd.DataFrame({'PassengerId':df_2['PassengerId'],
                      'Survived':estimator2.predict(X_subm)})

Subm3 = pd.DataFrame({'PassengerId':df_2['PassengerId'],
                      'Survived':estimator3.predict(X_subm)})

Subm4 = pd.DataFrame({'PassengerId':df_2['PassengerId'],
                      'Survived':grid_clf.predict(X_subm)})

Subm1.to_csv('Subm1.csv', sep = ',', index = False)
Subm2.to_csv('Subm2.csv', sep = ',', index = False)
Subm3.to_csv('Subm3.csv', sep = ',', index = False)
Subm4.to_csv('Subm4.csv', sep = ',', index = False)
