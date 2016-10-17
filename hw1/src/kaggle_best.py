# -*- coding: utf-8 -*-
"""
Created on Sun Oct 09 12:48:11 2016

@author: gary830317
"""

import csv
import pandas as pd

file_location = "./data/train.csv"
data = pd.read_csv(file_location) 

#初始weight
b = 2.0
w51 = 0.0
w41 = 0.1
w31 = -0.2
w21 = 0.0
w11 = 1.0
#當時測量的最佳參數:b=1.656  w51=-0.036  w41=0.35  w31=-0.472  w21=0.016  w11=1.076

T = 1000

l = 0.005

#將data中數值統一為浮點數
def data_float(i, j):
    f = float(data.iloc[i, j])
    return f


#main function



#重複做T次
for t in range(T):

    print t+1 
    loss = 0
    
    #adagrad累加一階微分值的平方的參數
    SG_b = 0.0
    SG_w51 = 0.0
    SG_w41 = 0.0
    SG_w31 = 0.0
    SG_w21 = 0.0
    SG_w11 = 0.0

    
    #檢測是不是數字
    for row in range(len(data)):

        G_b = 0.0
        G_w51 = 0.0
        G_w41 = 0.0
        G_w31 = 0.0
        G_w21 = 0.0
        G_w11 = 0.0
        
        if data.iloc[row,2] == "PM2.5":


            for col in range(8,26):
                
                f = b \
                  + w11 * data_float(row, col-1) \
                  + w21 * data_float(row, col-2) \
                  + w31 * data_float(row, col-3)\
                  + w41 * data_float(row, col-4)\
                  + w51 * data_float(row, col-5)
                  
                  
                #更新grant
                G_b = G_b - 2.0 * ((data_float(row, col)) - f) * 1.0 
                G_w51 = G_w51 - 2.0 * ((data_float(row, col)) - f) * data_float(row, col - 5)
                G_w41 = G_w41 - 2.0 * ((data_float(row, col)) - f) * data_float(row, col - 4)
                G_w31 = G_w31 - 2.0 * ((data_float(row, col)) - f) * data_float(row, col - 3)
                G_w21 = G_w21 - 2.0 * ((data_float(row, col)) - f) * data_float(row, col - 2)
                G_w11 = G_w11 - 2.0 * ((data_float(row, col)) - f) * data_float(row, col - 1)

                
                SG_b += ( 2.0 * ((data_float(row, col)) - f) * 1.0) ** 2
                SG_w51 += ( 2.0 * ((data_float(row, col)) - f) * data_float(row, col - 5)) ** 2
                SG_w41 += ( 2.0 * ((data_float(row, col)) - f) * data_float(row, col - 4)) ** 2
                SG_w31 += ( 2.0 * ((data_float(row, col)) - f) * data_float(row, col - 3)) ** 2
                SG_w21 += ( 2.0 * ((data_float(row, col)) - f) * data_float(row, col - 2)) ** 2
                SG_w11 += ( 2.0 * ((data_float(row, col)) - f) * data_float(row, col - 1)) ** 2

                
                loss += ((data_float(row, col)) - f)**2
            #觀察Loss
                
            #耕莘weight    
            b = b - ((l * G_b) / ((SG_b)** 0.5))
            w51 = w51 - ((l * G_w51) / ((SG_w51)** 0.5))
            w41 = w41 - ((l * G_w41) / ((SG_w41)** 0.5))
            w31 = w31 - ((l * G_w31) / ((SG_w31)** 0.5))
            w21 = w21 - ((l * G_w21) / ((SG_w21)** 0.5))
            w11 = w11 - ((l * G_w11) / ((SG_w11)** 0.5))

            
    print  "loss =", loss/4560        
    #觀察weight
    print b, w51, w41, w31, w21, w11
            



testfile_location = "./data/test_X.csv"
data_test = pd.read_csv(testfile_location) 

with open("./data/sampleSubmission.csv", 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['id', 'value'])
    t=0
    for row in range(len(data_test)):
        if data_test.iloc[row, 1] == "PM2.5":
            
            f = b \
            + w51 * float(data_test.iloc[row, 6]) \
            + w41 * float(data_test.iloc[row, 7]) \
            + w31 * float(data_test.iloc[row, 8]) \
            + w21 * float(data_test.iloc[row, 9]) \
            + w11 * float(data_test.iloc[row, 10]) 
            
            spamwriter.writerow(['id_%s'%t, f])
            t += 1
            
    print "saved"
    csvfile.close()






print "HW BEST"