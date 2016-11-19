
# coding: utf-8

# In[2]:

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





# In[3]:

def auto_encoder_model(data):
    batch_size = 50
    nb_classes = 10
    nb_epoch = 20

    
    input_img = Input(shape=(3072,))
    
    
    encoder=Dense(1024,activation='relu')(input_img)
    encoder_output=Dense(256)(encoder)

    decoder=Dense(1024,activation='relu')(encoder)
    decoder=Dense(3072,activation='sigmoid')(decoder)


    
    autoencoder = Model(input=input_img, output=decoder)
    encoder = Model(input=input_img, output=encoder_output)
    
    

    autoencoder.compile(loss='mean_squared_error',
                  optimizer='adam')
    
    if os.path.isfile(sys.argv[2])==False:
        autoencoder.save(sys.argv[2])
        
    autoencoder=load_model(sys.argv[2])


    autoencoder.fit(data, data,
                batch_size=batch_size,
                nb_epoch=nb_epoch,
                shuffle=True)
    
    autoencoder.save(sys.argv[2])
    encoder.save('encoder_model.h5')
    
    
    

    


# In[5]:

starttime1=time.time()
label_file_location=sys.argv[1]+'all_label.p'
unlabel_file_location=sys.argv[1]+'all_unlabel.p'
test_file_location=sys.argv[1]+'test.p'

label_data=pickle.load(open(label_file_location,'rb'))
unlabel_data=pickle.load(open(unlabel_file_location,'rb'))
test_data=pickle.load(open(test_file_location,'rb'))


# In[6]:

label_data_arr=np.array(label_data).astype('float32').reshape(5000,3072)/256
unlabel_data_arr=np.array(unlabel_data).astype('float32').reshape(45000,3072)/256




# In[14]:

t=0
data=np.concatenate((label_data_arr,unlabel_data_arr),axis=0)
while t <= 10:
    auto_encoder_model(data)
    t+=1


# In[ ]:



