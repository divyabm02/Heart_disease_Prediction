# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:08:30 2020

@author: id
"""

import pandas as pd
import numpy as np
import pickle

mydata=pd.read_csv("C:/Users/DIVYA/Desktop/College/CSV files/heart_disease.csv")
mydata.drop(['currentSmoker','education','diaBP','heartRate'],axis=1,inplace=True)

mean = mydata['glucose'].mean()
mydata['glucose'].fillna(mean, inplace=True)

mean = mydata['totChol'].mean()
mydata['totChol'].fillna(mean, inplace=True)

mean = mydata['BMI'].mean()
mydata['BMI'].fillna(mean, inplace=True)

mean = mydata['cigsPerDay'].mean()
mydata['cigsPerDay'].fillna(mean, inplace=True)

def missing_cat(series):
    mode=series.value_counts().index[0]
    series=series.fillna(mode)
    return series

for x in mydata:
    mydata[x]=missing_cat(mydata[x])
    
mydata.rename(columns={'male':'Gender'},inplace=True)

def vif_cal(input_data,dependent_col):
    import statsmodels.formula.api as sm
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]]
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.ols(formula="y~x",data=x_vars).fit().rsquared
        vif=round(1/(1-rsq),2)
        print (xvar_names[i], "VIF =" , vif)
        
vif_cal(input_data=mydata, dependent_col="TenYearCHD")

mydata.isnull().sum()

feature_cols = ['Gender', 'age', 'cigsPerDay','prevalentStroke', 'prevalentHyp',
                'diabetes', 'totChol', 'sysBP', 'BMI','glucose']

X = mydata[feature_cols]
Y = mydata.TenYearCHD

import sklearn.model_selection as ms
import sklearn.preprocessing as pre
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler

X = pre.minmax_scale(X)
X.shape

scaler = StandardScaler()
features_standardized = scaler.fit_transform(X)

from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
for i in enumerate(X):
    if X.dtype=='object':
        X=labe_encoder.fit_transform((X))
        
x_train,x_test,y_train,y_test = ms.train_test_split(X,Y,test_size=0.3,random_state=0)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

classifier=lm.LogisticRegression(random_state=0,class_weight=None,fit_intercept=True,intercept_scaling=1)
classifier.fit(x_train,y_train)
THRESHOLD = 0.16
y_pred = np.where(classifier.predict_proba(x_test)[:,1]>THRESHOLD,1,0)

pickle.dump(classifier,open("model.pkl","wb"))


        
