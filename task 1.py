#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation: Graduate Rotational Internship Program

# **<font size="5">Domain: Data Science And Business Analytics<br>**
#     
# <font size="3">Task1: To predict the percentage of a student based on number of study hours<br>
#     
# **<font size="3">Author: Reddigari Keerthi Reddy**

# In[96]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[97]:


#data extraction
data="http://bit.ly/w-data"
data=pd.read_csv(data)
print("data imported")
data.head(7)


# In[98]:


data.info()


# In[99]:


print("row",data.shape[0])
print("col",data.shape[1])


# In[100]:


data .dtypes


# In[101]:


data.describe()


# In[104]:


#plotting graph to show the distribution of scores
data.plot(x="Hours",y="Scores",style="o")
plt.title("Hours vs Scores")
plt.xlabel("study hours")
plt.ylabel("scores")
plt.show()


# In[105]:


#correlation of data
data.corr()


# **By analysing the above plot we can clearly observe that there is a linear relation between the hours studied and the percentage of the students.**  

# In[106]:


X=data.iloc[:,:-1].values
y=data.iloc[:,1].values


# In[107]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=40)


# In[108]:


#spliting data in to training and testing sets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[109]:


print(X_train, y_test)


# In[110]:


print(X_test, y_test)


# In[111]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
print(" completed")


# In[112]:


#regression line plotting
t_line=lr.coef_*X+lr.intercept_
#test data plotting
plt.scatter(X,y)
plt.plot(X,t_line);
plt.show()


# In[113]:


#test set prediction
print("y_test:")
print(y_test)
print("X_test:")
print(X_test)


# In[114]:


#scores prediction 
y_pred = lr.predict(X_test)
print("y_pred")
print(y_pred)


# In[115]:


#Actual vs Predict
dataframe=pd.DataFrame({"Actual":y_test,"Predict":y_pred})
dataframe


# In[116]:


# predicted score if a student study for 9.25 hours per day
hours=9.25
new_pred=lr.predict([[hours]])
print("study hours:{}".format(hours))
print("score prediction={}".format(new_pred[0]))


# In[117]:


from sklearn import metrics


# In[118]:


print("Mean Absolute Error:",metrics.mean_absolute_error(y_test,y_pred))


# In[119]:


print("Mean Squared Error:",metrics.mean_squared_error(y_test,y_pred))

