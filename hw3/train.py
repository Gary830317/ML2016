
# coding: utf-8

# In[313]:

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





# In[335]:

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

    # let's train the model .
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
    
    


# In[ ]:




# def cnn_model(label_data_arr, label):
#     model = Sequential()
#     model.add(Dense(input_dim=32*32,output_dim=800))
#     model.add(Activation('sigmoid'))
# 
#     model.add(Dense(800))
#     model.add(Activation('sigmoid'))
# 
#     model.add(Dense(8000))
#     model.add(Activation('sigmoid'))
# 
#     model.add(Dense(10))
#     model.add(Activation('softmax'))
# 
#     #model.summary()
# 
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     model.fit(label_data_arr,label,batch_size=50,nb_epoch=20)
#     

# In[315]:

def predict_to_labeled(label_data_arr, unlabel_data_arr, label):
    model=load_model(sys.argv[2])
    T1=model.predict(unlabel_data_arr)
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

    


# y=np.array([[0.1,0.2,0.3,0.81],[0.6,0.7,0.8,0.9]])
# print y
# z=np.array([0.6,0.7,0.8,0.9])
# #print any(y>0.8)
# print any(z>0.8)
# t=[]
# t.append(y.reshape(1,8).tolist())
# print t
# print np.array(t).reshape(2,4)

# print unlabel_data_arr[7]
# r=np.concatenate(unlabel_data_arr[7,8])
# print r
# 

# u=np.concatenate(label_data_arr,unlabel_data_arr[8].reshape(1,32,32,3),axis=0)
# print u.shape

# In[316]:

starttime1=time.time()
label_file_location=‘./‘+sys.argv[1]+’all_label.p’
unlabel_file_location=‘./‘+sys.argv[1]+’all_label.p’
test_file_location=./‘+sys.argv[1]+’test.p’

label_data=pickle.load(open(label_file_location,’rb’))
unlabel_data=pickle.load(open(unlabel_file_location,'rb'))
test_data=pickle.load(open(test_file_location,'rb'))


# In[337]:

label_data_arr=np.rollaxis(np.array(label_data).reshape(5000,3,32,32),1,4)
unlabel_data_arr=np.rollaxis(np.array(unlabel_data).reshape(45000,3,32,32),1,4)




# In[318]:

Z=[]
for i in range(10000):
    g=(np.array(test_data['data'][i]))
    Z=np.append(Z,g)
test_data_all=np.rollaxis(Z.reshape(10000,3,32,32),1,4)


# In[319]:

label=np.zeros((5000,10))
t=0
for i in range (10):
    for j in range(t,t+500):
        label[j][i]=1.
    t+=500
    
#print label



# print label_data_arr.shape[0]
# print label.shape[0]

# In[338]:

label_data_arr000=label_data_arr
unlabel_data_arr000=unlabel_data_arr
label000=label


# label_data_arr=label_data_arr000
# unlabel_data_arr=unlabel_data_arr000
# label=label000

# print label_data_arr.shape
# print label_data_arr000.shape
# print unlabel_data_arr.shape
# print unlabel_data_arr000.shape
# print label.shape
# print label000.shape
# 

# In[344]:

o=0
starttime2=time.time()
while o<=19 or label_data_arr.shape[0]>40000:
    print 'epoch:',o+1
    cnn_model(label_data_arr000, label000)
    o+=1
    print 'training time:',str((time.time()-starttime2)/60),'min'
    
while o<=34 or label_data_arr.shape[0]>40000:
    print 'epoch:',o+1 
    label_data_arr, label=predict_to_labeled(label_data_arr000, unlabel_data_arr , label000)
    cnn_model(label_data_arr, label)
    cnn_model(label_data_arr000, label000)
    o+=1
    print 'training time:',str((time.time()-starttime2)/60),'min'
   
    


# o=0
# while o<=10 or label_data_arr.shape[0]>40000:
#     print 'epoch:',o+1 
#     label_data_arr, label=predict_to_labeled(label_data_arr000, unlabel_data_arr , label000)
#     cnn_model(label_data_arr, label)
#     cnn_model(label_data_arr000, label000)
#     o+=1
#     print 'training time:',str((time.time()-starttime2)/60),'min'

# starttime=time.time()
# print starttime
# print 'training time:',starttime-time.time(),' sec'
# 

# print label_data_arr.shape

# unlabel_data_arr[5].shape
# u=np.concatenate([unlabel_data_arr,unlabel_data_arr[6].reshape(1,32,32,3)],axis=0)
# print u.shape

# a = np.array([[[1,2,3,4], [5,6,7,8], [9,10,11,12]],[[1,2,3,88], [5,6,7,88], [9,10,11,128]]])
# print a
# print'-----------'
# r=np.delete(a, [0,1], 0)
# print r
# 

# model.save_weights('model_semi_selflearning')

# print test_data_all

# model=load_model('semi_self_model.h5')
# result=model.predict(test_data_all)
# liste=[]
# for i in range(result.shape[0]):
#     clas = np.where(result[i] == result[i].max())
#     liste.append(list(clas)[0][0].tolist())
# print len(liste)
# 
# 

# print type(liste[9])

# with open('sys.argv[3]','wb')as csvfile:
#     spamwriter=csv.writer(csvfile,delimiter=',')
#     spamwriter.writerow(['ID','class'])
#     t=0
#     for row in range(len(liste)):
#         spamwriter.writerow([t,liste[t]])
#         t+=1
#     print 'saved'
#     csvfile.close()
# print 'total training time:',str((time.time()-starttime1)/60),'min'     

# dir(csv)
# 

# print label_data_arr.shape

# print(label_data.shape)

# label=np_utils.to_categorical(,10)
# print 123
# print(label.shape)
# 

# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')

# Y_train = np_utils.to_categorical(y_train, 10)
# Y_test = np_utils.to_categorical(y_test, 10)
# print Y_train.shape

# type(cifar10)
# print cifar10.shape

# type(label_data[2])
# 

# type(X_train)
# print X_train.shape
# 

# type(y_train)
# print y_train.shape
# #print y_train
# 

# data=[[[1,2,3],[1,2,3]],[[4,5,6],[4,5,6]]]
# d=np.array(data)
# print (d.shape)
# for x,y in d:
#     #print x
#     print type(y)
#     #print y
#     
#     
# 

# type(d)
# print label_data_arr.shape
# 

# a=[[1,2,3],[4,5,6]]
# random.shuffle(a)
# print a 

# e=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
# print e.reshape(2,10)
# print e.reshape(4,5)
# print e.reshape(2,2,5)

# l = np.array([[1,2,3], [4,3,1]])# Can be of any shape
# for i in range(l.shape[0]):
#     indices = np.where(l[i] == l[i].max())
#     print indices
