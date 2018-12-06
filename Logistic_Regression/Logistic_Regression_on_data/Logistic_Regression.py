# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 22:27:38 2018

@author: Atul
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


dataset = pd.read_csv('Social_Network_Ads.csv')

training_data , testing_data = np.split(dataset , [int(0.8 * len(dataset))])

#__________dataset for label 1__________

#__________select data features and label and convert into array and then normalise data__________

label_1 = training_data[training_data['Purchased'] == 1]

#__________To select label coloumn__________
training_data_label_1 = label_1.iloc[:,4:]

#__________To select feature coloumn__________
training_data_label_1_feature = label_1.iloc[:,2:4]

#__________To convert feature and label coloumn into aaray__________
training_data_label_1 = np.array(training_data_label_1)
training_data_label_1_feature = np.array(training_data_label_1_feature)

#__________normalise data label_1__________
training_data_label_1_feature = training_data_label_1_feature/training_data_label_1_feature.max(axis = 0)


#__________dataset for label 0__________

#__________select data features and label and convert into array and then normalise data__________

label_0 = training_data[training_data['Purchased'] == 0]

#__________To select label coloumn__________
training_data_label_0 = label_0.iloc[:,4:]

#__________To select feature coloumn__________
training_data_label_0_feature = label_0.iloc[:,2:4]

#__________To convert feature and label coloumn into aaray__________
training_data_label_0 = np.array(training_data_label_0)
training_data_label_0_feature = np.array(training_data_label_0_feature)

#__________normalise data label_0__________
training_data_label_0_feature = training_data_label_0_feature/training_data_label_0_feature.max(axis = 0)

#__________random weights__________
weights = np.random.randn(1,training_data_label_0_feature.shape[1])
weights_default = np.random.randn(1)


#__________sigmoid function__________

def sigmoid(weights_default , weights , training_data_label_1_feature , training_data_label_0_feature):
    sigmoid_label_1 = np.zeros(len(label_1))
    sigmoid_label_1 = sigmoid_label_1.reshape(len(label_1), 1)
    for i in range(0 , len(label_1)):
        sigmoid_label_1[i] = 1.0/(1+ np.exp(-(weights_default + np.dot(weights , training_data_label_1_feature[i]))))
        
    sigmoid_label_0 = np.zeros(len(label_0))
    sigmoid_label_0 = sigmoid_label_0.reshape(len(label_0) , 1)
    for j in range(0 , len(label_0)):
        sigmoid_label_0[j] = 1.0/(1+ np.exp(-(weights_default + np.dot(weights , training_data_label_0_feature[j]))))
    
    return sigmoid_label_1 , sigmoid_label_0

#__________likelihodd function__________
    
def likelihood(weights_default,weights,training_data_label_1_feature , training_data_label_0_feature):
    sigmoid_label_1 , sigmoid_label_0 = sigmoid(weights_default,weights,training_data_label_1_feature , training_data_label_0_feature)
    
    likelihood_sum = (np.sum(np.log(sigmoid_label_1)) + np.sum(np.log(1-sigmoid_label_0)))
    
    return likelihood_sum
    
    

#__________derivative__________
    
def derivative(weights_default,weights,training_data_label_1_feature , training_data_label_0_feature):
    sigmoid_label_1, sigmoid_label_0 = sigmoid(weights_default,weights,training_data_label_1_feature , training_data_label_0_feature)
    derivative_sigmoid_1 = 1 - sigmoid_label_1
    derivative_sigmoid_0 = sigmoid_label_0
    
    #__________calculate Fi(1-sigmoid) + Fi(sigmoid)__________
    
    
    slope = (np.sum(training_data_label_1_feature*derivative_sigmoid_1 , axis = 0)) - (np.sum(training_data_label_0_feature*derivative_sigmoid_0 , axis = 0))
#    first_der = first_der.reshape(1,2)
    # for theta 0
    intercept = (np.sum(derivative_sigmoid_1)) - (np.sum(derivative_sigmoid_0))
    
    return slope,intercept

#__________gradient__________
    
def gradient(weights_default,weights,training_data_label_1_feature , training_data_label_0_feature):
    
    slope_old = weights
    intercept_old = weights_default
    learning_rate = 0.001
    epsilon = 0.001
    i = 0
    
    while(True):
        
        likelihood_sum = likelihood(intercept_old,slope_old,training_data_label_1_feature , training_data_label_0_feature)
        print("Likelihood_sum", likelihood_sum)
        
        slope_derivative, intercept_derivative = (derivative(intercept_old,slope_old,training_data_label_1_feature , training_data_label_0_feature))
        print("Slope_der", slope_derivative)
        print("Intercept_der", intercept_derivative)
        
        slope_new = slope_old + learning_rate*slope_derivative
        intercept_new = intercept_old + learning_rate*intercept_derivative
        print("Slope_new" , slope_new, "Slope_old", slope_old)
        print("Intercept_new" , intercept_new, "Intercept_old", intercept_old)
        
#        weights_default_new = weights_default_old - learning_rate*sec_der
        
        
    
        if ((np.abs(np.sum(np.abs(slope_new)) - np.sum(np.abs(slope_old)))  < epsilon) and (np.abs(intercept_new) - np.abs(intercept_old) < epsilon)):
            print("slope_diff" ,np.abs(np.sum(np.abs(slope_new)) - np.sum(np.abs(slope_old))))
            print("intercepet_diff",np.abs(intercept_new) - np.abs(intercept_old) )
            break
        
        print("slope_diff" ,np.abs(np.sum(np.abs(slope_new)) - np.sum(np.abs(slope_old))))
        print("intercepet_diff",np.abs(intercept_new) - np.abs(intercept_old) )
        
        slope_old = slope_new
        intercept_old = intercept_new
        
        i+=1
        
        print(i)
        
    return slope_new , intercept_new
    

#print(derivative(weights_default,weights,training_data_label_1_feature , training_data_label_0_feature))
#print(gradient(weights_default,weights,training_data_label_1_feature , training_data_label_0_feature))




#__________Testing__________

    











