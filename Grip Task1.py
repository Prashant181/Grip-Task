#!/usr/bin/env python
# coding: utf-8

# 
# # Prediction using supervised Machine Learning

# Given Task :- What will be predicted score if a student studies for 9.25 hrs/ day?

# In[2]:


# importing required libraries


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


# In[4]:


# data reading process
df=pd.read_csv('http://bit.ly/w-data')
df.head()


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


df.info()


# In[8]:


df.describe()


# # data Pre visualization

# In[9]:


plt.scatter(df['Hours'],df['Scores'])
plt.title('Hours vs Percentage')  
plt.xlabel('Hours')  
plt.ylabel('Score')  
plt.grid()
plt.show()


# From the graph above, we can clearly see that there is a positive 
# linear relation between the number of hours studied and percentage of score.

# In[ ]:





# Next Divide the data into inputs and output.

# In[12]:


x=df.drop(['Scores'],axis=1)
y=df['Scores']


# # We split our data into training and testing sets.

# Here we do spliting of our data set in 30/70 part , 30% part is to be used as test dataset on our module 70% part is use as to train our module

# In[13]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=30)


# # Applying Simple Learning Regression Algorithm

# In[14]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)


# In[15]:


coeficient=model.coef_
coeficient


# In[16]:


intercept=model.intercept_
intercept


# Yhat=3.1671+x*9.7433

# In[17]:


line=intercept+x*coeficient


# In[18]:


plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# In[19]:


ypred=model.predict(xtest)
ypred


# In[20]:


pd.DataFrame({'Actual':ytest,'Predict':ypred})


# # Data visualization

# In[21]:


plt.scatter(xtest,ytest, color ='black')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.grid()
plt.title("Actual Values Plot")
plt.show()

plt.scatter(xtest,ypred ,color ='red')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.grid()
plt.title('Predicted values Plot')
plt.show()


# # Accuracy of the model

# In[22]:


model.score(xtrain,ytrain)


# In[23]:


model.score(xtest,ytest)


# In[24]:


model.score(xtest,ypred)


# # Task

# In[26]:


a=pd.DataFrame({'Hours':[9.25]})
a


# In[27]:


model.predict(a)


# Predicted score of 9.25 Hours is 93.2928

# In[ ]:




