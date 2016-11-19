
# coding: utf-8

# In[5]:

import pickle
import numpy as np
import random
import pandas as pd
import csv
import h5py
import time
import os
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model


# In[6]:

test_file_location=sys.argv[1]+'test.p'
test_data=pickle.load(open(test_file_location,'rb'))
Z=[]
for i in range(10000):
    g=(np.array(test_data['data'][i]))
    Z=np.append(Z,g)
test_data_all=np.rollaxis(Z.reshape(10000,3,32,32),1,4)



# In[7]:

model=load_model(sys.argv[2])
result=model.predict(test_data_all)
liste=[]
for i in range(result.shape[0]):
    clas = np.where(result[i] == result[i].max())
    liste.append(list(clas)[0][0].tolist())
print len(liste)


# In[8]:

with open(sys.argv[3],'wb')as csvfile:
    spamwriter=csv.writer(csvfile,delimiter=',')
    spamwriter.writerow(['ID','class'])
    t=0
    for row in range(len(liste)):
        spamwriter.writerow([t,liste[t]])
        t+=1
    print 'saved'
    csvfile.close()
print 'total training time:',str((time.time()-starttime1)/60),'min'     


# In[ ]:



