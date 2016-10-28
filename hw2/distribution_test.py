# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:38:06 2016

@author: gary830317
"""

#%%

import csv
import pandas as pd
import numpy as np
import math
import sys
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


testfile_location = sys.argv[2]
data_test = pd.read_csv(testfile_location,encoding='big5',header=None) 

data_test.columns=['id']+range(1,58)
data_test[55]=data_test[55]/10
data_test[56]=data_test[56]/50
data_test[57]=data_test[57]/1000

pc0=2447/4001
pc1=1554/4001

g= np.loadtxt('model1.txt')
cov_matrix=g.reshape(57,57)

h= np.loadtxt('model2.txt')
everage0=np.array(h[0:57])
everage1=np.array(h[57:114])


with open(sys.argv[3], 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['id', 'label'])
    t=0
    for row in range(len(data_test)):
        x = data_test.iloc[row,range(1,58)].get_values().tolist()
        f = p(x,everage0,everage1,cov_matrix)
        if f > 0.5:
            label = 1
        else:
            label = 0
        spamwriter.writerow([t+1, label])
        t += 1
            
    print "saved"
    csvfile.close()
