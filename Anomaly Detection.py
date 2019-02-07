# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:25:04 2019

@author: paras
"""

import pandas as pd
import numpy as np
import random
dataset = pd.read_csv('creditcard.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 30].values


#separating positive examples
positive_x = []
positive_y = []
negative_x = []
negative_y = []
for i in range(len(y)):
    if y[i] == 1 :
        positive_x.append(X[i])
        positive_y.append(y[i])
    else :
        negative_x.append(X[i])
        negative_y.append(y[i])

#splitting dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        negative_x, negative_y, test_size=0.2, random_state=42)

#Adding the positive examples to test set
X_test.extend(positive_x)
y_test.extend(positive_y)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Calculating mean for each feature
Mean_X_train = []
for i in range(X_train.shape[1]):
  Mean_X_train.append(X_train[:, i].mean()) 

#Calculating standard deviation for each feature
std_X_train = []
for j in range(X_train.shape[1]):
  std_X_train.append(np.std(X_train[:, j], axis=0)) 
  
#Calculating the probability of each exaple in test set
import math
y_pred = []
def norm(x, mean, std):
    variance = float(std)**2
    denom = (2*math.pi*variance)**.5
    numer = math.exp(-(float(x)-float(mean))**2/(2*variance))
    return numer/denom
  
for i in range(len(X_test)):
     p = 1
     for j in range(len(Mean_X_train)):
         p *= normpdf(X_test[i, j], Mean_X_train[j], std_X_train[j])
     y_pred.append(p)

#converting the probability to 0s and 1s depending on threshold value
y_out = []
for i in range(len(y_pred)):
    if y_pred[i] < 9.914460408105776e-21 :
        y_out.append(1) 
    else:
        y_out.append(0)
        
#checking efficiency with confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_out)