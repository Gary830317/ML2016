
# coding: utf-8

# In[23]:

import csv
import sklearn
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import sys


# In[24]:

f=open('stopword.txt','r')
stopword=[]
for word in f:
    #print word
    stopword=word.decode('utf8').split(' ')
#print type(stopword[0])
#print stopword[10]
stopcha=['\"',':','?','.','(',')','~','*','_','[',']',',',';','&','%','=','+','-','#','/','\n',' ']
f.close()


# In[25]:

#make a doc-term matrix
f=open(sys.argv[1]+'title_StackOverflow.txt','r')
a=f.readlines()

title_list=[]
for lines in a:
    title=''
    #print lines
    lines=lines.split(' ')
    #print lines
    for word in lines:
        worda=''
        for cha in word:
            
            if cha not in stopcha:
                worda+=cha
        #print word
        #print worda
        #word=str(unicode(word, errors='ignore'))
        #print word
        if worda.lower() not in stopword:
            #print'y'
            title=title+str(worda).lower()+' '
    #print title
    title_list.append(title)
    #print title_list   
vectorizer=TfidfVectorizer()
tfidf=vectorizer.fit_transform(title_list)
print tfidf.shape

terms=vectorizer.get_feature_names()
#print terms


# print type(title)

# pca=PCA(n_components=10)
# reduse=pca.fit_transform(tfidf.todense())
# #print reduse
# 

# In[26]:

svd=TruncatedSVD(18)
normalizer=Normalizer(copy=False)
lsa=make_pipeline(svd,normalizer)
reduse=lsa.fit_transform(tfidf)


# print reduse[[11726,8]]

# In[27]:

labels=KMeans(25).fit_predict(reduse)
f=open(sys.argv[1]+'check_index.csv','r')
lines = f.readlines()
prediction=[]
for i in range(1,len(lines)):
    line=lines[i].split(',')
    #print line[1],line[2]
    ans=1 if labels[int(line[1])]==labels[int(line[2])] else 0
    prediction.append(ans)
    
#print prediction
    


# print prediction[0:100]

# for index,item in enumerate(prediction):
#     if item=='2':
#         prediction[index]='0'
# 

# In[28]:

with open(sys.argv[2],'w') as csvfile:
    spamwriter=csv.writer(csvfile,delimiter=',')
    spamwriter.writerow(['ID','Ans'])
    t=0
    for i in range(len(prediction)):
        spamwriter.writerow([t,prediction[i]])
        t+=1
    print 'saved'
    csvfile.close()
    


# In[ ]:



