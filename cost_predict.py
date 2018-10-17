# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 11:49:53 2018

@author: SHRIKRISHNA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("insurance.csv")

df.describe()

df.info()

data_grouped = df.groupby(['smoker','sex']).agg({'charges':'sum', 'sex':'count'})
#data_grouped.index=[1,2,3,4]
data_grouped


df.dtypes

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
encoder.fit(df['sex'].drop_duplicates())
df['sex'] = encoder.transform(df['sex'])
encoder.fit(df['smoker'].drop_duplicates())
df['smoker'] = encoder.transform(df['smoker'])
#X = df.iloc[:,:-1].values
#encoder = LabelEncoder()
#X[:,5] = encoder.fit_transform(X[:,5])
#onehotencoder = OneHotEncoder(categorical_features = [5])
#X = onehotencoder.fit_transform(X).toarray()
#to use onehotencoder we have to take new varible X and consider our colume in that using iloc

data1 = pd.get_dummies(df['region'], prefix = 'region')
df1 = pd.concat([df,data1], axis = 1).drop (['region'], axis = 1)



print(df.head(2))

#df1.dtypes()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

'''y = df1[:,[5]]
X = df1[:, [0,1,2,3,6,7,8,9]]
lin_reg = LinearRegression()

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.25, random_state = 21)
lin_reg.fit(train_X, train_y)

pred_y = lin_reg.predict(test_X) 

rmse = np.sqrt(mean_squared_error(test_y, pred_y))
print ("RMSE : %f" % (rmse))'''

y= df1['charges']
X = df1.drop(['charges'], axis=1)
lin_reg=LinearRegression()
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25, random_state=21)
lin_reg.fit(train_X,train_y)
pred_y=lin_reg.predict(test_X)
rmse = np.sqrt(mean_squared_error(test_y, pred_y))
print("RMSE: %f" % (rmse))


import statsmodels.formula.api as sm

X_opt = X = df1.drop(['charges'], axis=1)
regressor_OLS = sm.OLS (endog = y, exog = X_opt.values).fit()
regressor_OLS.summary()

X_opt = X = df1.drop(['charges','sex'], axis=1)
regressor_OLS = sm.OLS (endog = y, exog = X_opt).fit()
regressor_OLS.summary()


def backwardElimination(x,sl):
    numVars = len(x[0])
    for i in range (0, numVars):
        regressor_OLS = sm.OLS(y,x).fit()
        maxVar = max(regressor_OLS.pvalues)
        
        if maxVar > sl:
            for j in range(0, numVars - i):
                if(regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x,j,1)
    regressor_OLS.summary()
    
    return x

SL = 0.05
X_opt = df1.drop(['charges'], axis=1)
X_modeled = backwardElimination (X_opt.values, SL)


y= df1['charges']
X = df1.drop(['charges','sex'], axis=1)

lin_reg = LinearRegression()

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.25, random_state = 21)
lin_reg.fit(train_X, train_y)

pred_y = (lin_reg.predict(test_X))

rmse = np.sqrt(mean_squared_error(test_y, pred_y))
print ("RMSE : %f" % (rmse))

from sklearn.metrics import f1_score
print("Avg F1-score : %.4f" %f1_score(test_y, pred_y, average = 'weighted'))

from sklearn.metrics import jaccard_similarity_score as jc
jc(test_y, pred_y)
