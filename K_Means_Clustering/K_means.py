# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 22:55:17 2018

@author: Atul
"""

import numpy as np
import pandas as pd
import random

raw_data = pd.read_csv('test.csv')
data = raw_data.iloc[:,5:]



mean_old_1 = data.iloc[random.randint(0, len(data))]
mean_old_2 = data.iloc[random.randint(0, len(data))]


data = np.array(data)

mean_old_1 = np.array(mean_old_1)
mean_old_2 = np.array(mean_old_2)



i = 0

while(True):
    
    cluster_1 = []
    cluster_2 = []
    
    for i in range(0 , len(data)):
        
        d1 = np.dot((mean_old_1 - data[i]) , (mean_old_1 - data[i]).T)
        d2 = np.dot((mean_old_2 - data[i]) , (mean_old_2 - data[i]).T)
        
        if d1 < d2:
            cluster_1.append(data[i])
        else:
            cluster_2.append(data[i])
        
    mean_new_1 = np.average(cluster_1)
    mean_new_2 = np.average(cluster_2)
    
    if (np.sum(mean_old_1) - np.sum(mean_new_1) == 0) and (np.sum(mean_old_2) - np.sum(mean_old_2) == 0):
        
        break
    
    mean_old_1 = mean_new_1
    mean_old_2 = mean_new_2
    
    i+=1
    
    