
# coding: utf-8

# In[7]:

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





# In[8]:

def cnn_model(label_data_arr, label):
    batch_size = 50
    nb_classes = 10
    nb_epoch = 10

    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='same',
                            input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    #model.summary()

    
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    if os.path.isfile(sys.argv[2])==False:
        model.save(sys.argv[2])
        
    model=load_model(sys.argv[2])
    
    model.fit(label_data_arr, label,
                batch_size=batch_size,
                nb_epoch=nb_epoch,
                shuffle=True)
    
    model.save(sys.argv[2])
    
    


# In[9]:

def predict_to_labeled(label_data_arr, unlabel_data_arr, label):
    models=load_model(sys.argv[2])
    T1=models.predict(unlabel_data_arr)
    #print T1.shape[0]
    #print type(T1)
    w=[]
    ss=[]
    sss=[]
    for i in range(T1.shape[0]):
        if any(T1[i]>0.95):
            w.append(i)
            sss.append(unlabel_data_arr[i].reshape(1,3072).tolist())
            clas=list(np.where(T1[i]==T1[i].max()))[0][0]
            h=np.zeros(10)
            h[clas]=1
            ss.append(h.tolist())
    new_label_data_arr=np.concatenate((label_data_arr,np.array(sss).reshape(len(w),32,32,3)),axis=0)
    new_label=np.concatenate((label,np.array(ss).reshape(len(w),10)),axis=0)
    #unlabel_data_arr=np.delete(unlabel_data_arr,w,0)
    #print 'unlabel data:',unlabel_data_arr.shape[0]
    return [new_label_data_arr, new_label]

    


# In[10]:

starttime1=time.time()
label_file_location=sys.argv[1]+'all_label.p'
unlabel_file_location=sys.argv[1]+'all_unlabel.p'
test_file_location=sys.argv[1]+'test.p'

label_data=pickle.load(open(label_file_location,'rb'))
unlabel_data=pickle.load(open(unlabel_file_location,'rb'))


# In[11]:

label_data_arr=np.rollaxis(np.array(label_data).reshape(5000,3,32,32),1,4)
unlabel_data_arr=np.rollaxis(np.array(unlabel_data).reshape(45000,3,32,32),1,4)




# In[13]:

label=np.zeros((5000,10))
t=0
for i in range (10):
    for j in range(t,t+500):
        label[j][i]=1.
    t+=500
    
#print label



# In[14]:

label_data_arr000=label_data_arr
unlabel_data_arr000=unlabel_data_arr
label000=label


# In[16]:

print label_data_arr.shape
print unlabel_data_arr.shape


# In[18]:

o=0
starttime2=time.time()
while o<=19 or label_data_arr.shape[0]>40000:
    print 'epoch:',o+1
    cnn_model(label_data_arr000, label000)
    o+=1
    print 'training time:',str((time.time()-starttime2)/60),'min'
    
while o<=39 or label_data_arr.shape[0]>40000:
    print 'epoch:',o+1 
    label_data_arr, label=predict_to_labeled(label_data_arr000, unlabel_data_arr , label000)
    cnn_model(label_data_arr, label)
    cnn_model(label_data_arr000, label000)
    o+=1
    print 'training time:',str((time.time()-starttime2)/60),'min'
   
    


# In[ ]:



