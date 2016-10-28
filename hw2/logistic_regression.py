# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 18:12:22 2016

@author: gary830317
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:04:22 2016

@author: gary830317
"""

#%%
import csv
import pandas as pd
import random 
import numpy as np
import math


#%%
def sigmoid_function(z):
    return 1.0/(1.0+math.exp(-z))
    
    
#%%
# X是dim=57的vector
def prediction(X,w,b):
    return sigmoid_function(np.dot(X,w)+b)


#%%
# 
def logistic_regression(train,w,b,learningrate):
    G_w =np.zeros(57)
    G_b = 0.0
    s_loss = 0.0
    
    for i in range(0,len(train)):
        tem = train.iloc[i,range(1,58)].get_values().tolist()
        f = prediction(tem,w,b)
        if f == 0.0:
            f = 0.00001
        if f == 1.0:
            f = 0.99999
        real = train.iloc[i,58] 
        s_loss -= (real*math.log(f)+(1-real)*math.log(1-f))
        G_w0 = np.dot((real - f),tem)
        G_w = np.array(G_w)-np.array(G_w0)
        G_b -= (real - f)
        
#        print s_loss
    w0 = np.dot(learningrate,G_w)
    w = np.array(w)-np.array(w0)
    b -= learningrate*G_b
    loss = s_loss/len(train)
    print loss
    return [w,b]

#%%

traindata_location='sys.argv[1]'

train=pd.read_csv(traindata_location,encoding='big5',header=None)
train.columns=['id']+range(1,58)+['label']
train[55]=train[55]/10
train[56]=train[56]/50
train[57]=train[57]/1000



#initial
w = np.random.rand(57)/100
b = np.random.rand(1)
learningrate = 0.0005
T = 1000

for t in range(T):
    
    t += 1
    print t
    w,b=logistic_regression(train,w,b,learningrate)

h = open('sys.argv[2].txt', 'w')
h.write(' '.join("%f" % i for i in w.tolist()+b.tolist()))
h.close()


