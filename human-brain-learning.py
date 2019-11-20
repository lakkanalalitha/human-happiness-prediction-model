# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:26:03 2019

@author: lkumari
"""

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pickle
'''
df = pd.read_excel('datasets/human-brain-learning.xls', header=[1])

print(df.head())



X = df.loc[:, df.columns != 'happy'].values
y = df.loc[:, df.columns == 'happy'].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

scaled_data.mean(axis = 0)
scaled_data.std(axis = 0)

df.corr(method ='pearson')
# the columns using kendall method 
df.corr(method ='kendall') 


test_size = 0.22
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)

from sklearn.linear_model import LogisticRegression

m = LogisticRegression(
    penalty='l1',
    C=0.1
)
m.fit(X_train, Y_train)
m.predict_proba(X_test)[:,1]

result = m.score(X_test, Y_test)
print("Accuracy: %.3f%%" % (result*100.0))

####################################################################
        #K-fold Cross Validation + Logistic Regression
###################################################################           
model = LogisticRegression(solver='lbfgs')
kfold = model_selection.KFold(n_splits=10)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print("Mean Accuracy: %.3f%%" % (results.mean()*100.0))
print("Std Deviation: %.3f%%" % (results.std()*100.0))

kf = KFold(n_splits=5)
test_accuracy = []
train_accuracy = []
tempTrain = 0
tempTest = 0
for nbrOfFolds,(train_index, test_index) in enumerate(kf.split(X)):
    ## Splitting the data into train and test
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test  = y[train_index], y[test_index]
    
    ##fit the data into the model
    model.fit(X_train,Y_train)
    
    ##predicting the values on the fitted model using train data
    predTrain = model.predict((X_train))
    
    #adding the accuracy
    tempTrain = tempTrain + accuracy_score(Y_train,predTrain)
    
    ##predict the values on the fitted model using test data
    predTest = model.predict((X_test))
    
    #adding the accuracy
    tempTest = tempTest + accuracy_score(Y_test,predTest)
    
##Calculating the train accuracy
print(f'Number of folds is{nbrOfFolds+1}') 
train_accuracy.append(tempTrain*1.0/(nbrOfFolds+1))

##Calculating the test accuracy
test_accuracy.append(tempTest*1.0/(nbrOfFolds+1))
print("(train,test) accuracy = ",tempTrain*1.0/(nbrOfFolds+1), tempTest*1.0/(nbrOfFolds+1))

'''
#==============================================================================
# pickling
#==============================================================================
'''
with open('human-life-balancing-indicator.pickle','wb') as f:
    pickle.dump(model, f)
'''


pickle_in = open('human-life-balancing-indicator.pickle','rb')
model = pickle.load(pickle_in)

 
unseen_x = np.array([ 4,  9,  1,  4,  6])
prob = model.predict([unseen_x])

confidence_measure = model.predict_proba([unseen_x])[0, prob]

print("Unseen Data Accuracy on K-Fold: %.3f%%" % (confidence_measure*100.0))
pred_target = prob[0]


if(pred_target == 1):
    print("You are more happier person ")
else:
    print("You are less happier person !!! ")


####################################################################
        #Leave One Out Cross Validation + Logistic Regression
#################################################################### 
'''
looc_model = LogisticRegression()
loocv = model_selection.LeaveOneOut()
loocv_results = model_selection.cross_val_score(looc_model, X, y, cv=loocv)

print("Mean Accuracy: %.3f%%" % (loocv_results.mean()*100.0))
print("Std Deviation: %.3f%%" % (loocv_results.std()*100.0))   
looc_model.fit(X_train, Y_train)  
 
unseen_x = np.array([ 4,  9,  1,  4,  6])
loo_prob = looc_model.predict([unseen_x])

loo_confidence_measure = looc_model.predict_proba([unseen_x])[0, loo_prob]

print("Unseen Data Accuracy on Loo: %.3f%%" % (loo_confidence_measure*100.0))      
        
import statsmodels.api as sm

logit = sm.Logit(y, X)
result = logit.fit_regularized()
'''
