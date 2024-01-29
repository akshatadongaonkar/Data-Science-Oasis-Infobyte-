#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# In[2]:


df=pd.read_csv('Iris.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


data=df.groupby('Species')


# In[7]:


df['Species'].unique()


# In[8]:


#2. visualizing the dataset
plt.boxplot(df['SepalLengthCm'])


# In[9]:


plt.boxplot(df['SepalWidthCm'])


# In[10]:


plt.boxplot(df['PetalLengthCm'])


# In[11]:


plt.boxplot(df['PetalWidthCm'])


# In[12]:


sns.heatmap(df.corr())


# In[13]:


#3. Data Preparation
df.drop('Id',axis=1,inplace=True)


# In[14]:


sp={'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}


# In[15]:


df.Species=[sp[i] for i in df.Species]


# In[16]:


df


# In[17]:


X=df.iloc[:,0:4]


# In[19]:


X


# In[20]:


y=df.iloc[:,4]


# In[21]:


y


# In[22]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# In[23]:


#Traning Model
model=LinearRegression()


# In[24]:


model.fit(X,y)


# In[25]:


model.score(X,y) #coef of prediction


# In[26]:


model.coef_


# In[27]:


model.intercept_


# In[28]:


#Making Prediction
y_pred=model.predict(X_test)


# In[29]:


#Model Evolution
print("Mean squared error: %.2f" % np.mean((y_pred - y_test) ** 2))


# In[ ]:




