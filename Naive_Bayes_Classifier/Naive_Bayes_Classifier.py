# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 22:37:09 2018

@author: Atul
"""
import pandas as pd
import numpy as np
import random
import scipy.stats as S
import math


#def multivariate_prob():
    
#    prob =1/(math.sqrt(2*np.pi))**10*

uncleaned_data = pd.read_csv('data.csv')
uncleaned_data.dropna(axis= 1 , inplace= True)
uncleaned_data.replace(to_replace='B', value=0, inplace=True)
uncleaned_data.replace(to_replace='M', value=1, inplace=True)
uncleaned_data = np.array(uncleaned_data)
label = uncleaned_data[:,1:2]
uncleaned_data = uncleaned_data[:,2:]

cleaned_data=np.zeros(np.shape(uncleaned_data))


# Normalizing the data
for i in range(30):
    col=uncleaned_data[:,i]
    col=col-np.min(col)
    col=col/np.max(col)
    cleaned_data[:,i]=col



# selecting benign and malignant label

malignant = np.where(label == 1)
malignant = malignant[0]

benign = np.where(label == 0)
benign = benign[0]


# -----PCA-----#

def PCA(cleaned_data):

    cleaned_data_cov = np.cov(cleaned_data , rowvar = 0)
    
    eigen_val , eigen_vec = np.linalg.eig(cleaned_data_cov)
    
    eigen_val_argsort = np.argsort(eigen_val)
    
    sorted_eigen_val = np.zeros(len(eigen_val))
    sorted_eigen_vec = np.zeros(np.shape(eigen_vec))
    
    
    for i in range(len(eigen_val)):
        sorted_eigen_val[len(eigen_val) - i - 1] = eigen_val[eigen_val_argsort[i]]
        sorted_eigen_vec[:,len(eigen_vec) - i - 1] = eigen_vec[:,eigen_val_argsort[i]]
        
    count = 0
    weight = np.zeros(len(eigen_val))
    for i in range(len(eigen_val)):
        
        weight[i] = (count + sorted_eigen_val[i])/np.sum(sorted_eigen_val)
        count = count + sorted_eigen_val[i]
        
    var = 0.80
    
    a = np.where(weight >= var)
    
    a = a[0]
    a = a[0]
    imp_eigen_vec = sorted_eigen_vec[:,:a+1]
    
    new_data = np.matmul(cleaned_data , imp_eigen_vec)

    return new_data

new_data = PCA(cleaned_data)

# --- Seprate the data----#

malignant_data=new_data[malignant,:]
benign_data=new_data[benign,:]


malignant_cov = np.cov(malignant_data , rowvar= 0)
benign_cov = np.cov(benign_data , rowvar= 0)

malignant_mean = np.mean(malignant_data , axis = 0)
benign_mean = np.mean(benign_data , axis = 0)

testing_val = new_data[random.randint(0, len(new_data)),:]

#--Testing --#

#-- Malignant--#
malignant_prob = S.multivariate_normal.pdf(testing_val , malignant_mean , malignant_cov)
prob_m = len(malignant_data)/len(new_data)
final_prob_m = malignant_prob * prob_m

#-- Benign--#

benign_prob = S.multivariate_normal.pdf(testing_val , benign_mean , benign_cov)
prob_b = len(benign_data)/len(new_data)
final_prob_b = benign_prob * prob_b


if final_prob_b > final_prob_m:
    print("Tumor is Benign")
else:
    print("Tumor is Malignant")