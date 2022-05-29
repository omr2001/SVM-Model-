#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import cv2 as cv 
import glob 
import random
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from skimage.feature import hog


# In[2]:


file=r'C:\Users\Adel\Downloads\train\train\*.jpg'
glob.glob(file)


# In[3]:


Full_Train_Data=[]
for i in glob.glob(file):
    Resized_Data=cv.resize(cv.imread(i,1),(128,64))
    Full_Features,hog_image=hog(Resized_Data,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,multichannel=True)
    if 'cat' in i :
        Full_Train_Data.append([Full_Features,0])
    else :
        Full_Train_Data.append([Full_Features,1])


# In[4]:


Full_Train_Data=pd.DataFrame(Full_Train_Data)
Train_Data=pd.DataFrame()
Test_Data=pd.DataFrame()
Train_Data=Train_Data.append(Full_Train_Data.iloc[0:1000])
Train_Data=Train_Data.append(Full_Train_Data.iloc[12500:13500])
Test_Data=Test_Data.append(Full_Train_Data.iloc[1000:1100])
Test_Data=Test_Data.append(Full_Train_Data.iloc[13500:13600])
len(Train_Data)
Train_Data
print(Test_Data)


# In[5]:


type(Train_Data)


# In[6]:


Train_Data = Train_Data.sample(frac = 1)
Train_Data


# In[7]:


X_Train=Train_Data[0]
X_Train=list(X_Train)
Y_Train=Train_Data[1]
Y_Train=list(Y_Train)
print(len(X_Train))
print(len(Y_Train))


# In[8]:


print(type(X_Train))
X_Train=np.array(X_Train)
Y_Train=np.array(Y_Train)
rbf_svc = svm.SVC().fit(X_Train, Y_Train)
Train_Res=rbf_svc.predict(X_Train)
print("The Training Accuracy:",rbf_svc.score(X_Train,Y_Train))


# In[9]:


Test_Data = Test_Data.sample(frac = 1)
Test_Data


# In[10]:


X_Test=Test_Data[0]
X_Test=list(X_Test)
Y_Test=Test_Data[1]
Y_Test=list(Y_Test)


# In[11]:


print(type(X_Test))
X_Test=np.array(X_Test)
Y_Test=np.array(Y_Test)
res=rbf_svc.predict(X_Test)
accuracy=np.mean(res==Y_Test)
print("The Acc is ",accuracy)
print("The Testing Accuracy:",rbf_svc.score(X_Test,Y_Test))


# In[ ]:




