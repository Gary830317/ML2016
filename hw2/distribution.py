# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 23:25:21 2016

@author: gary830317
"""

#%%
import csv
import pandas as pd
import random 
import numpy as np
import math


#%%
def everage(data):
    a = np.zeros(57)
    for i in range(len(data)):
        tem = np.array(data.iloc[i,range(1,58)].get_values())
        a += tem/len(data)
    return a
#%%
def cov_matrix(data,u):
    data1 = data.loc[:,range(1,58)]
    s = np.zeros(shape=(57,57))
    for i in range(len(data1)):
        a = (np.array(data1.iloc[i,:])-u)*(np.array(data1.iloc[i,:])-u).reshape(57,1)
        s+=a
    return s

#%%
    

def gausian(x,everage,cov_matrix):
    pi=math.pi
    p=((1/(2*pi))**(57/2)) * (1/(np.linalg.det(cov_matrix)**0.5))*np.exp(-0.5*(np.dot(np.dot((np.array(x)-everage),(cov_matrix**-1)),(np.array(x)-everage))))
    return p

#%%
def p(x,everage0,everage1,cov_matrix):
    f=(gausian(x,everage0,cov_matrix)*pc0)/(gausian(x,everage0,cov_matrix)*pc0+gausian(x,everage1,cov_matrix)*pc1)
    return f

#%%
traindata_location='sys.argv[1]'

train=pd.read_csv(traindata_location,encoding='big5',header=None)
train.columns=['id']+range(1,58)+['label']
train[55]=train[55]/10
train[56]=train[56]/50
train[57]=train[57]/1000


everage = train.groupby('label', as_index=False, group_keys=False, sort=False).apply(everage) 
everage0 = np.array(everage[0])
everage1 = np.array(everage[1])

cov_matrix0 = cov_matrix(train[(train.label==0)],everage0)
cov_matrix1 = cov_matrix(train[(train.label==1)],everage1)
cov_matrix = (2447*cov_matrix0 + 1554*cov_matrix1)/4001

pc0=2447/4001
pc1=1554/4001

with file('model1.txt', 'w') as h:
    np.savetxt(h,cov_matrix)
    h.close()

h = open('model2.txt', 'w')
h.write(' '.join('%f'%i for i in everage0.tolist() + everage1.tolist()))
h.close()

print 'saved'









