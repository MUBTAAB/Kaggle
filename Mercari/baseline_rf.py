# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
print('Modules')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Any results you write to the current directory are saved as output.
print('Datasets')
df_train = pd.read_csv('../input/train.tsv', sep='\t')
df_test = pd.read_csv('../input/test.tsv', sep='\t')

print('Cathegory names')
def GetCat(df):
    df['catlen'] = [np.NaN if pd.isnull(i) else len(i.split('/')) for i in df['category_name']]
    df['cat0'] = [np.NaN if pd.isnull(i) else i.split('/')[0] for i in df['category_name']]
    df['cat1'] = [np.NaN if pd.isnull(i) else i.split('/')[1] for i in df['category_name']]
    df['cat2'] = [np.NaN if pd.isnull(i) else i.split('/')[2] for i in df['category_name']]
    
GetCat(df_train)
GetCat(df_test)


print('Title cleaning')
def NameClean(df):
    df['name'] = df['name'].fillna('missing')

    lname = []
    dset = zip(df['name'], df['brand_name'], df['cat0'], df['cat1'], df['cat2'])
    for name, brand, cat0, cat1, cat2 in dset:
        nname = name.lower()
        for i in [brand, cat0, cat1, cat2]:
            if pd.isnull(i) == False:
                nname = nname.replace(i,'')
        lname.append(nname)
    df['name'] = lname

NameClean(df_train)
NameClean(df_test)


print('Vectorizing_names')
vectorizer = CountVectorizer(input='content',binary=True, lowercase = True ,max_features = 100)
vectorizer.fit(df_train['name'].values)

def VectorizeName(df):
    tfMatrix = vectorizer.transform(df['name']).toarray()
    dfMatrix = pd.DataFrame(tfMatrix)
    dfMatrix.columns = vectorizer.get_feature_names()[:]
    dfMatrix.columns = ['ntoken_'+str(i) for i in dfMatrix.columns]
    return(df.join(dfMatrix))

df_train = VectorizeName(df_train)
df_test = VectorizeName(df_test)


# Dataclean code from https://www.kaggle.com/cbrogan/xgboost-example-python/code  
# We'll impute missing values using the median for numeric columns and the most
# common value for string columns
print('Modelling dataprep')
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

feature_columns_to_use = ['item_condition_id',
                        'shipping',
                        'brand_name',
                        'catlen',
                        'cat0', 
                        'cat1', 
                        'cat2']
                        
#feature_columns_to_use.extend(df_train.columns[11:])
nonnumeric_columns = ['brand_name','cat0', 'cat1', 'cat2']

# Join the features from train and test together before imputing missing values,
# in case their distribution is slightly different
big_X = df_train[feature_columns_to_use].append(df_test[feature_columns_to_use])
big_X_imputed = DataFrameImputer().fit_transform(big_X)

# XGBoost doesn't (yet) handle categorical features automatically, so we need to change
# them to columns of integer values.
# See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more
# details and options
le = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

# Prepare the inputs for the model
train_X = big_X_imputed[0:df_train.shape[0]].as_matrix()
test_X = big_X_imputed[df_train.shape[0]::].as_matrix()
train_y = df_train['price']

print('Modelling')
from sklearn.ensemble import RandomForestRegressor
predictor = RandomForestRegressor().fit(train_X, train_y)
print('Predicting')
predictions = predictor.predict(test_X)

print('Submitting')
df_test['price'] = predictions
submit = df_test[['test_id','price']]
submit.to_csv('submit_base1.csv',index=False)
