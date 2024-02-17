#!/usr/bin/env python
# coding: utf-8
# TASKNO:01
author: prajakta chavan
batch :15THfeb-15H MARCH 2024
domain: data science
project name: IRIS FLOWER CLASSIFICATION USING MACHINE LEARNING
# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os
sns.set(style="white",color_codes=True)


# In[79]:


#irisflowerclassification path 
df=pd.read_csv(r'C:\Users\Dell\Desktop\irisflowerclassification.csv')


# In[80]:


#print irisflowerclassification 
df


# In[10]:


#print 1st data for irisflowerclassification
df.head()


# In[11]:


#pandas and dataframe and overview project
df.info()


# In[12]:


#Species value integerformats and print data 
df['Species'],categories=pd.factorize(df['Species'])
df.head()


# In[13]:


df.describe


# In[14]:


df.isna().sum()


# In[15]:


df.columns


# In[61]:


df["Species"].value_counts()


# In[17]:


#describe data
df.describe()


# In[18]:


y=df.iloc[:,4]
y


# In[19]:


x=df.iloc[:,0:4]
x


# In[20]:


print(x.shape)
print (y.shape)


# In[21]:


plt.boxplot(df['SepalLengthCm'])


# In[42]:


plt.boxplot(df['SepalWidthCm'])


# In[43]:


plt.boxplot(df['PetalLengthCm'])


# In[44]:


plt.boxplot(df['PetalWidthCm'])


# In[22]:


df['SepalLengthCm'].hist()


# In[23]:


df['SepalWidthCm'].hist()


# In[24]:


df['PetalLengthCm'].hist()


# In[30]:


df['PetalLengthCm'].hist()


# In[25]:


df['PetalWidthCm'].hist()


# In[75]:


#graph point :
sns.pairplot(df,hue='Species')


# In[93]:


#import the data 
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[96]:


from sklearn.model_selection import train_test_split
#train-70
#test=30
x=df.drop(columns=['Species'])
y=df['Species']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.50)


# In[97]:


#LogisticRegression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[98]:


model.fit(x_train,y_train)


# In[99]:


model.score(x_test,y_test)
print("Accuracy:",model.score(x_test,y_test)*100)


# In[100]:


#kneighborsclassifier
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()


# 

# In[101]:


model.fit(x_train,y_train)


# In[102]:


print("Accuracy:",model.score(x_test,y_test)*100)


# In[103]:


#decisin making 
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()


# In[104]:


model.fit(x_train,y_train)


# In[105]:


print("Accuracy:",model.score(x_test,y_test)*100)


# In[ ]:




