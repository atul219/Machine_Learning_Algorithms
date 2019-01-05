# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 23:35:44 2018

@author: Atul
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt 
import math

Df = pd.read_csv("test.csv")


Df.head()

DfCopy = Df.iloc[:,5:]

DfCopy.head()

#np.sum(DfCopy.iloc[0:5],axis = 1)

DfCopy.shape

# TREE CODE

class TreeNode:
    
    def __init__(self):
        
        self.RowIndices = None
        self.FeatureName = None
        self.Lvl = None
        self.Median = None
        self.Min_Distance = None

#---- FUNCTIONS----
        
def euclidiean(table):
     
    sq_distance=table**2
    rmse=np.sqrt(np.sum(sq_distance,axis=1))
    return rmse
    
def scaledEculeadean(table,weights):
    sq_distance=table**2
    weight_sq_distance=sq_distance*weights
    rmse=np.sqrt(np.sum(weight_sq_distance,axis=1))
    return rmse
    
   
def check(cluster):
    
    distance = Testing_Example - DfCopy.iloc[NodesListBFS[cluster].RowIndices] 
    rmse=euclidiean(distance)
   # rmse=scaledEculeadean(distance,[1,2,2,1])
    min_distance=min(rmse)
    index=np.where(rmse==min_distance)
    index=distance.iloc[index]
    
    return min_distance,index 


# ----CODE----       
        
DfCopyFeatures = list(DfCopy.columns)

RootNode = TreeNode ()

RootNode.RowIndices = DfCopy.index

NodesListBFS = []

NodesListBFS.append(RootNode)

Level = 0

for Feature in DfCopyFeatures:
    
    ParentNodeIndex = ((2**Level) - 1)
    
    NumberOfParentNodes = 2**Level
    
    print("Level # {}, Number of Parent Nodes = {}".format(Level,NumberOfParentNodes))
     
    
    for i in range(0,NumberOfParentNodes):
        
        LeftChildNewNode = TreeNode ()
        
        DataInParentNode = pd.DataFrame(DfCopy.iloc[NodesListBFS[ParentNodeIndex].RowIndices][Feature])
        
        
        
        print("Parent Node at Level # {}, Number of Data Points = {}".format(Level,len(NodesListBFS[ParentNodeIndex].RowIndices)))
        
        LeftChildNewNode.RowIndices = DataInParentNode[DataInParentNode[Feature] <= DataInParentNode[Feature].median()].index
      
        NodesListBFS[ParentNodeIndex].Median = DataInParentNode[Feature].median()
        
        NodesListBFS[ParentNodeIndex].FeatureName = Feature
        
        # min distance
        #NodesListBFS[ParentNodeIndex].Min_Distance = np.sum(DataInParentNode.iloc[])
        
        #LeftChildNewNode.FeatureName = Feature
        
        LeftChildNewNode.Lvl = (Level + 1)
        
        print("Left Child at Level # {}, Number of Data Points = {}, Feature Name = {}".format(LeftChildNewNode.Lvl,len(LeftChildNewNode.RowIndices), LeftChildNewNode.FeatureName))
        
        
        RightChildNewNode = TreeNode ()
        
        RightChildNewNode.RowIndices = DataInParentNode[DataInParentNode[Feature] > DataInParentNode[Feature].median()].index
        
        #RightChildNewNode.FeatureName = Feature
        
        RightChildNewNode.Lvl = (Level + 1)
        
        
        print("Right Child at Level # {}, Number of Data Points = {}, Feature Name = {}".format(RightChildNewNode.Lvl,len(RightChildNewNode.RowIndices), RightChildNewNode.FeatureName))
        
        NodesListBFS.append(LeftChildNewNode)
        
        NodesListBFS.append(RightChildNewNode)
        
        print("Currently at Level # {}, Feature Name is {}, Inserted Two Child Nodes at Level # {}".format(Level,NodesListBFS[ParentNodeIndex].FeatureName,LeftChildNewNode.Lvl))
        
        ParentNodeIndex += 1
        
        print("Going to the new parent node at the same level now.......\n\n")
    
    Level += 1
    
    print("Sorry ! No Parent Nodes left to traverse in the current level")
    
    print("Going to Change the level now........\n\n\n\n")
    
print("Binary Search Tree Creation in Completed.")

#print(NodesListBFS[0].Median)
#print(NodesListBFS[0].FeatureName)

#len(NodesListBFS)

# --TESTING--

DfCopyFeatures

Testing_Example = DfCopy.iloc[random.randint(0,len(DfCopy))]

Pi = 0

for Feature in DfCopyFeatures:
    
    if Testing_Example[Feature] <= NodesListBFS[Pi].Median :
        
        Pi = (2*Pi) + 1
        
    else:
        
        Pi = (2*Pi) + 2
        
#print(NodesListBFS[Pi])
        
#print(NodesListBFS[Pi].RowIndices)
        
#print(DfCopy.iloc[NodesListBFS[Pi].RowIndices])
        
#print(DfCopy.iloc[131])
#print(Testing_Example)
        
      

Level_left_node = 2**Level-1  
Level_mid_node = Level_left_node + 2**(Level-1)-1
if (Pi == Level_left_node) or (Pi == Level_mid_node+1) :
    l_dis,l_index = check(Pi) 
    r_dis,r_index = check(Pi+1)
    if l_dis < r_dis :
        min_dis = l_dis
        index = l_index
    else:
        min_dis = r_dis
        index = r_index
elif (Pi == Level_mid_node) or (Pi == 2**(Level+1)-2):
    r_dis,r_index = check(Pi)
    l_dis,l_dis = check(Pi-1)
    if l_dis < r_dis :
        min_dis = l_dis
        index = l_index
    else:
        min_dis = r_dis
        index = r_index
    
else:
    dis,index = check(Pi)
    r_dis,r_index = check(Pi+1)
    l_dis,l_index = check(Pi-1)
    if min(r_dis,l_dis,dis) == r_dis:
        min_dis = r_dis
        index = r_index
    elif min(r_dis,l_dis,dis) == l_dis:
        min_dis = l_dis
        index = l_index
    else:
        min_dis = dis
        index = index
        
print(min_dis)
print(index)