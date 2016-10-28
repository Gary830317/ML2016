# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 13:39:14 2016

@author: gary830317
"""

#%%
import csv
import pandas as pd
import random 
import numpy as np
import math
import sys

#%%
def sigmoid_function(z):
    return 1.0/(1.0+math.exp(-z))
    
    
#%%
# X是dim=57的vector
def prediction(X,w,b):
    return sigmoid_function(np.dot(X,w)+b)

#%%
testfile_location = sys.argv[2]
data_test = pd.read_csv(testfile_location,encoding='big5',header=None) 

data_test.columns=['id']+range(1,58)
data_test[55]=data_test[55]/10
data_test[56]=data_test[56]/50
data_test[57]=data_test[57]/1000

g= np.loadtxt(sys.argv[1])
w=g[0:57]
b=g[57]


with open(sys.argv[3], 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['id', 'label'])
    t=0
    for row in range(len(data_test)):
        tem1 = data_test.iloc[row,range(1,58)].get_values().tolist()
        f = prediction(tem1,w,b)
        if f > 0.5:
            label = 1
        else:
            label = 0
        spamwriter.writerow([t+1, label])
        t += 1
            
    print "saved"
    csvfile.close()
