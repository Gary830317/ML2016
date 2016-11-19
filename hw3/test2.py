
# coding: utf-8

# In[1]:

import pickle
import numpy as np
import random
import pandas as pd
import csv
import h5py
import time
import os
import sys
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Model


# In[ ]:

starttime1=time.time()
label_file_location=sys.argv[1]+'all_label.p'
unlabel_file_location=sys.argv[1]+'all_unlabel.p'
test_file_location=sys.argv[1]+'test.p'

label_data=pickle.load(open(label_file_location,'rb'))
unlabel_data=pickle.load(open(unlabel_file_location,'rb'))
test_data=pickle.load(open(test_file_location,'rb'))


# In[4]:

label_data_arr=np.array(label_data).astype('float32').reshape(5000,3072)/256
unlabel_data_arr=np.array(unlabel_data).astype('float32').reshape(45000,3072)/256

data=np.concatenate((label_data_arr,unlabel_data_arr),axis=0)


# In[3]:

encoder=load_model(sys.argv[2])
T1=encoder.predict(data)
kmeans=KMeans(10).fit(T1)
prediction=kmeans.predict(encoder.predict(test_data_all)).reshape(10000,1)


# In[ ]:

with open(sys.argv[3],'wb')as csvfile:
    spamwriter=csv.writer(csvfile,delimiter=',')
    spamwriter.writerow(['ID','class'])
    t=0
    for row in range(len(prediction)):
        spamwriter.writerow([t,prediction[t][0]])
        t+=1
    print 'saved'
    csvfile.close()
print 'total training time:',str((time.time()-starttime1)/60),'min'  

