import pandas as pd
import re
import numpy as np

mydata=pd.read_csv("C:/Users/DIVYA/Downloads/proj_flaskdm (1)/proj_flask/adult-training.csv")

mydata['income'] = mydata['income'].apply(lambda x: 1 if x== ' >50K' else 0)
mydata['gender'] = mydata['gender'].apply(lambda x: 1 if x== 'male' else 0)

#combining native_country
def country(x):
    if re.search('United-States', x):
        return 'US'
    else:
        return 'not-US'
mydata['native_country']=mydata.native_country.apply(lambda x: x.strip()).apply(lambda x: country(x))

mydata['native_country'] = mydata['native_country'].apply(lambda x: 1 if x=='US' else 0)
#combinig workclass
def workclass(x):
    if re.search('Private', x):
        return 'Private'
    elif re.search('Self', x):
        return 'selfempl'
    elif re.search('gov', x):
        return 'gov'
    else:
        return 'other class'
mydata['work_class']=mydata.work_class.apply(lambda x: x.strip()).apply(lambda x: workclass(x))

#combinig marital_Status
def maritalstatus(x):
    if re.search('Married', x):
        return 'Married'
    elif re.search('Never-married', x):
        return 'Never-married'
    else:
        return 'Single'
mydata['marital_status']=mydata.marital_status.apply(lambda x: x.strip()).apply(lambda x: maritalstatus(x))

#combining education
def education(x):
    if re.search('Bachelors', x):
        return 'Bachelors'
    elif re.search('Some-College', x):
        return 'others'
    elif re.search('Masters', x):
        return 'Higher-education'
    elif re.search('Doctorate', x):
        return 'Higher-education'
    elif re.search('Prof-school',x):
        return 'Professional'
    elif re.search('Assoc-voc',x):
        return 'Professional'
    elif re.search('Assoc-acdm',x):
        return 'Professional'
    else:
        return 'Primary'
mydata['education']=mydata.education.apply(lambda x: x.strip()).apply(lambda x: education(x))


mydata = pd.concat([mydata,pd.get_dummies(mydata['work_class'])],axis=1)
mydata = pd.concat([mydata,pd.get_dummies(mydata['marital_status'])],axis=1)
mydata = pd.concat([mydata,pd.get_dummies(mydata['native_country'],prefix="country")],axis=1)
mydata = pd.concat([mydata,pd.get_dummies(mydata['education'])],axis=1)
mydata = pd.concat([mydata,pd.get_dummies(mydata['race'])],axis=1)
mydata = pd.concat([mydata,pd.get_dummies(mydata['occupation'])],axis=1)
mydata = pd.concat([mydata,pd.get_dummies(mydata['relationship'])],axis=1)

mydata.columns 

from sklearn import preprocessing
encoder=preprocessing.LabelEncoder()
mydata=mydata.apply(encoder.fit_transform)

encoder.fit(mydata['native_country'])
encoder.transform(mydata['native_country'])

mydata.drop(['gender','fnlwgt','marital_status','work_class','native_country','education','race','occupation','relationship'],axis=1,inplace=True)

#Labels and featureSet columns
columns = mydata.columns.tolist()
columns = [c for c in columns if c not in ['income']]
target = 'income'

X = mydata[columns]
Y = mydata[target]
mydata.columns

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=232426)

x_train.shape,x_test.shape,y_train.shape,y_test.shape

from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression(max_iter=10000,class_weight='balanced',fit_intercept=True,solver='lbfgs')
logreg.fit(x_train,y_train)

y_pred = logreg.predict(x_test)

logreg.fit(x_train,y_train)
THRESHOLD = 0.6
y_pred = np.where(logreg.predict_proba(x_test)[:,1]>THRESHOLD,1,0)

import pickle
with open('logreg.pkl', 'wb') as fid:
    pickle.dump(logreg, fid,2) 

#Create a Dataframe with only the dummy variables
cat = mydata.drop('income',axis=1)
index_dict = dict(zip(cat.columns,range(cat.shape[1])))

with open('cat', 'wb') as fid:
    pickle.dump(index_dict, fid,2)