#!/usr/bin/env python
# coding: utf-8

# In[50]:


# !pip install imbalanced-learn
import librosa 
import soundfile as sf
import pandas as pd
import numpy as np
import os
import zipfile
import imblearn
from imblearn.over_sampling import SMOTE


# In[2]:


cwd1='C:/Users/Areeb Hassan/Downloads/PAI_VolunteerTask'


# In[3]:


cwd1


# In[98]:


mfcc=[]
Chromagram=[]
labels=[]
for i in os.listdir(cwd1):
    #print(i)
    cwd2= os.path.join(cwd1,i)
    
    for j in os.listdir(cwd2):
        if('.wav' in j):
            #print(j)
            soundFile= os.path.join(cwd2,j)
            x,sr= librosa.load(soundFile)
            
            mfccfeatures=librosa.feature.mfcc(y=x,sr=sr)
            m= mfccfeatures.flatten().tolist()
            mfcc.append(m)
            print(mfccfeatures)
            
            chromagram=librosa.feature.chroma_stft(y=x,sr=sr,hop_length=512)
            c=chromagram.flatten().tolist()
            Chromagram.append(c)
            
            labels.append(i)


# In[99]:


size=[]
for i in mfcc:
    size.append(len(i))
    
maximum=max(size)

for j in mfcc:
    while (len(j)<maximum):
        j.append(0)
        
MfccData=pd.DataFrame(mfcc)
MfccData.to_csv('C:/Users/Areeb Hassan/Downloads/MfccData.csv')


# In[100]:


size=[]
for i in Chromagram:
    size.append(len(i))
    
maximum=max(size)

for j in Chromagram:
    while (len(j)<maximum):
        j.append(0)
        
ChromagramData=pd.DataFrame(Chromagram)
ChromagramData.to_csv('C:/Users/Areeb Hassan/Downloads/ChromagramData.csv')


# In[101]:


df=pd.concat([MfccData,ChromagramData],axis=1)


# In[ ]:


# size=[]
# for i in mel:
#     size.append(len(i))
    
# maximum=max(size)

# for j in mel:
#     while (len(j)<maximum):
#         j.append(0)
        
# melData=pd.DataFrame(mel)
# melData.to_csv('C:/Users/Areeb Hassan/Downloads/melData.csv')
# melData


# In[78]:


# !pip install sklearn
from sklearn.model_selection import train_test_split


# In[79]:


from sklearn.preprocessing import LabelEncoder
import pickle


# In[103]:


le = LabelEncoder()
label=le.fit_transform(labels)
y=pd.DataFrame(label)


# In[81]:


smote = SMOTE()
X_sm,y_sm = smote.fit_resample(df,y)


# In[106]:


X_sm.shape


# In[107]:


X_train,X_test,y_train,y_test= train_test_split(X_sm , y_sm ,test_size=0.2,random_state=24)


# In[109]:


#ClassRoom's Notebook
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
pickle.dump(clf,open('tree.pkl','wb'))


# In[110]:


pred=clf.predict(X_test)
print (pred)
clf.score(X_test,y_test)


# In[111]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 
print ("Accuracy : " , accuracy_score(y_test,pred)*100)  
print("Report : \n", classification_report(y_test, pred))
print("F1 Score : ",f1_score(y_test, pred, average='macro')*100)


# In[108]:


#ClassRoom's Notebook
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf = clf.fit(X_train, y_train)
pickle.dump(clf,open('knn.pkl','wb'))
pred=clf.predict(X_test)
print (pred)
clf.score(X_test,y_test)
print ("Accuracy : " , accuracy_score(y_test,pred)*100)  
print("Report : \n", classification_report(y_test, pred))
print("F1 Score : ",f1_score(y_test, pred, average='macro')*100)


# In[116]:


# https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
from sklearn import svm
clf=svm.SVC()
clf = clf.fit(X_train, y_train)
pickle.dump(clf,open('svc.pkl','wb'))
pred=clf.predict(X_test)
print (pred)
clf.score(X_test,y_test)
print ("Accuracy : " , accuracy_score(y_test,pred)*100)  
print("Report : \n", classification_report(y_test, pred))
print("F1 Score : ",f1_score(y_test, pred, average='macro')*100)


# In[115]:


# https://www.datacamp.com/community/tutorials/adaboost-classifier-python
from sklearn.ensemble import AdaBoostClassifier
clf=AdaBoostClassifier()
clf = clf.fit(X_train, y_train)
pickle.dump(clf,open('ada.pkl','wb'))
pred=clf.predict(X_test)
print (pred)
clf.score(X_test,y_test)
print ("Accuracy : " , accuracy_score(y_test,pred)*100)  
print("Report : \n", classification_report(y_test, pred))
print("F1 Score : ",f1_score(y_test, pred, average='macro')*100)


# In[114]:


# https://www.programcreek.com/python/example/87301/sklearn.naive_bayes.GaussianNB
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf = clf.fit(X_train, y_train)
pickle.dump(clf,open('gnb.pkl','wb'))
pred=clf.predict(X_test)
print (pred)
clf.score(X_test,y_test)
print ("Accuracy : " , accuracy_score(y_test,pred)*100)  
print("Report : \n", classification_report(y_test, pred))
print("F1 Score : ",f1_score(y_test, pred, average='macro')*100)


# In[113]:


# https://www.geeksforgeeks.org/ml-extra-tree-classifier-for-feature-selection/
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
extra_tree_forest = ExtraTreesClassifier()
extra_tree_forest.fit(X_train, y_train)
pickle.dump(extra_tree_forest,open('etf.pkl','wb'))
pred=extra_tree_forest.predict(X_test)
print (pred)
extra_tree_forest.score(X_test,y_test)
print ("Accuracy : " , accuracy_score(y_test,pred)*100)  
print("Report : \n", classification_report(y_test, pred))
print("F1 Score : ",f1_score(y_test, pred, average='macro')*100)


# In[ ]:




