# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:25:04 2019

@author: paras
"""

import pandas as pd
import matplotlib.pyplot as plt
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

X_test.extend(positive_x)
y_test.extend(positive_y)

random.shuffle(X_test)
random.shuffle(y_test)

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
import scipy.stats
y_pred = []
for i in range(len(X_test)):
    p = 1
    for j in range(len(Mean_X_train)):
        p *= scipy.stats.norm(Mean_X_train[j], std_X_train[j]).pdf(X_test[i, j])
    y_pred.append(p)

#Calculating the probability of each exaple in training set
import scipy.stats
y_train_pred = []
for i in range(len(X_train)):
    p = 1
    for j in range(len(Mean_X_train)):
        p *= scipy.stats.norm(Mean_X_train[j], std_X_train[j]).pdf(X_train[i, j])
    y_train_pred.append(p)

#probabilities of all positive examples  
indices_of_positives =[]
for i in range(len(y_test)):
    if y_test[i] == 1:
        indices_of_positives.append(i)
    
probs_of_positives = []    
for i in indices_of_positives:
    probs_of_positives.append(y_pred[i])

#Defining y_pred for a value of epsilon
for i in range(len(y_pred)):
    if y_pred[i] < 3.914460408105776e-14 :
        y_pred[i] = 1
    else:
        y_pred[i] = 0

#checking efficiency with confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
        