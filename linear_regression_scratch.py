# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 23:16:40 2018

@author: Atul
"""

import pandas as pd
import numpy as np
import random


raw_data = pd.read_csv('USA_Housing.csv')

# spliting training data and testing data

training_data , testing_data = np.split(raw_data , [int(0.8 * len(raw_data))])

# Normalize Training Data

training_data_feature = training_data.iloc[:,0:4]
training_data_label = training_data.iloc[:,5:6]

training_data_feature = np.array(training_data_feature)
training_data_label = np.array(training_data_label)

training_data_feature = training_data_feature/training_data_feature.max(axis = 0)
training_data_label = training_data_label/training_data_label.max(axis = 0)

# Normalize Testing Data

testing_data_feature = testing_data.iloc[:,0:4]
testing_data_label = testing_data.iloc[:,5:6]

testing_data_feature = np.array(testing_data_feature)
testing_data_label = np.array(testing_data_label)

testing_data_feature = testing_data_feature/testing_data_feature.max(axis = 0)
testing_data_label = testing_data_label/testing_data_label.max(axis = 0)

no_features = len(training_data_feature.T)

weights = np.random.randn(1,4)
weights_default = np.random.randn(1)


def predict(weights_default,weights,data):
    z = np.zeros(len(data))
    for i in range(0 , len(data)):
        z[i] = weights_default + np.dot(weights , data[i])
    z = z.reshape(len(data) , 1)
    return z

def cost(weights_default,weights,data,data_label):
    diff = np.square(predict(weights_default,weights,data) - data_label)
    cost = np.mean(diff)/2
    
    return cost

def derivative(weights_default,weights,training_data_feature):
    slope = np.mean((predict(weights_default,weights,training_data_feature) - training_data_label)*training_data_feature , axis = 0)
    intercept = np.mean((predict(weights_default,weights,training_data_feature) - training_data_label))
    
    return slope , intercept


def gradient(weights_default,weights,training_data_feature):
    
    learning_rate = 0.01
    slope_old = weights
    intercept_old = weights_default
    i = 0
    
    while(True):
        
        slope_derivative , intercept_derivative = derivative(intercept_old,slope_old,training_data_feature)
        
        slope_new = slope_old - learning_rate*slope_derivative
        intercept_new = intercept_old - learning_rate*intercept_derivative
        
        if (np.abs(np.sum(np.abs(slope_new)) - np.sum(np.abs(slope_old))) < 0.01 and np.abs(np.abs(intercept_new) - np.abs(intercept_old)) < 0.01):
            print('slope_diff' , np.abs(np.sum(np.abs(slope_new)) - np.sum(np.abs(slope_old))))
            print('intercept_diff', np.abs(np.abs(intercept_new) - np.abs(intercept_old)))
            print('We have got Best slope and intercept')
            break
        
        slope_old = slope_new
        intercept_old = intercept_new
        
        i+=1
        print('iter' , i)
        
        
    return slope_new , intercept_new


#print('Slope_new', 'Intercept_new' , gradient(weights_default,weights,training_data_feature))

slope_new , intercept_new = gradient(weights_default,weights,training_data_feature)

    
# Testing
print('----------Testing----------') 
print('Cost on Testing Data',cost(intercept_new,slope_new,testing_data_feature , testing_data_label) )
 