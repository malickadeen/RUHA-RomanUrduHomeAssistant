#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pickle
import sklearn
import librosa 
import soundfile as sf
import pandas as pd
import numpy as np
import os
import zipfile
import imblearn
from imblearn.over_sampling import SMOTE


# In[77]:


def labels(i):
    print(i)
    if(i==0):
        return "A.C"
    if(i==1):
        return "Bulb"
    if(i==2):
        return "Gana"
    if(i==3):
        return "T.V"


# In[78]:


knn=pickle.load(open('knn.pkl','rb'))
etf=pickle.load(open('etf.pkl','rb'))
gnb=pickle.load(open('gnb.pkl','rb'))
ada=pickle.load(open('ada.pkl','rb'))
svc=pickle.load(open('svc.pkl','rb'))
tree=pickle.load(open('tree.pkl','rb'))


# In[79]:


cwd1='C:/Users/Areeb Hassan/Downloads/sample.wav'


# In[80]:


x,sr= librosa.load(cwd1)


# In[81]:


mfccfeatures=librosa.feature.mfcc(y=x,sr=sr)
m= mfccfeatures.flatten().tolist()
while(len(m)<12161):
        m.append(0)
        
# MfccData=pd.DataFrame(m)


# In[82]:


MfccData=pd.DataFrame(m).T


# In[83]:


MfccData


# In[84]:


knnPrediction=knn.predict(MfccData)
etfPrediction=etf.predict(MfccData)
gnbPrediction=gnb.predict(MfccData)
adaPrediction=ada.predict(MfccData)
svcPrediction=svc.predict(MfccData)
treePrediction=tree.predict(MfccData)


# In[86]:


knnPrediction=labels(knnPrediction)
etfPrediction=labels(etfPrediction)
gnbPrediction=labels(gnbPrediction)
adaPrediction=labels(adaPrediction)
svcPrediction=labels(svcPrediction)
treePrediction=labels(treePrediction)


# In[ ]:




