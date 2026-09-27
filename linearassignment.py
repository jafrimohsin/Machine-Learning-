#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("Linear_regression_gpt.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[10]:


f=df.corr()


# In[11]:


f


# In[17]:


sns.heatmap(f,annot=True,cmap="coolwarm")
plt.title("Heatmap")


# In[21]:


sns.scatterplot(x="Experience",y="Salary",data=df)
plt.title("Experience vs Salary")


# In[23]:


X=df.drop("Salary",axis=1)


# In[24]:


y=df["Salary"]


# In[25]:


X


# In[26]:


y


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[32]:


X_train


# In[33]:


from sklearn.linear_model import LinearRegression


# In[34]:


model = LinearRegression()


# In[35]:


model.fit(X_train,y_train)


# In[36]:


model.intercept_


# In[37]:


model.coef_


# In[38]:


y_pred = model.predict(X_test)


# In[39]:


y_pred


# In[40]:


y_test


# In[42]:


sns.scatterplot(x=y_test,y=y_pred)
plt.ylabel("predicted Salary")


# In[46]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,root_mean_squared_error


# In[48]:


mean_absolute_error(y_test,y_pred)


# In[49]:


mean_squared_error(y_test,y_pred)


# In[50]:


root_mean_squared_error(y_test,y_pred)


# In[51]:


r2_score(y_test,y_pred)


# In[53]:


mean_squared_error(y_test,y_pred)**0.5


# In[55]:


df.head(1)


# In[60]:


new_emp=[[30,5,17,45,7]]
prediction=model.predict(new_emp)


# In[61]:


prediction


# In[64]:


new_emp=[[23,0,17,20,2]]
prediction=model.predict(new_emp)


# In[65]:


prediction


# In[ ]:




