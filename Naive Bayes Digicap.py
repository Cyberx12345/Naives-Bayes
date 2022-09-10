#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libraries
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# In[3]:


#loading the data
data = pd.read_csv(r"C:\Users\Benjamin\Desktop\icon\Downloads\golfdataset.csv")


# In[4]:


data


# In[7]:


#encoding data: string to numeric
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data['Outlook'] = le.fit_transform(data.Outlook.values)
data['Temp'] = le.fit_transform(data.Temp.values)
data['Humidity'] = le.fit_transform(data.Humidity.values)
data['Windy'] = le.fit_transform(data.Windy.values)
data['PlayGolf'] = le.fit_transform(data.PlayGolf.values)


# In[9]:


#After changing or transforming the data from categorical to Numerical
data


# In[18]:


#splitting data in Train and Test
#x = predictors which are features
#y = labels or answers 

labels = data['PlayGolf']
features = data.drop('PlayGolf' , axis=1)
#Test Size and random state to determine the outcome 
X_train,X_test,Y_train,Y_test = train_test_split(features , labels , train_size=0.2 , random_state=0)


# In[19]:


X_train


# In[20]:


X_test


# In[24]:


Y_train


# In[25]:


Y_test


# In[35]:


#calling the GAussian Bayes 
# training the model to make predictions
gnb = GaussianNB()
gnb.fit(X_train , Y_train)


# In[36]:


#Dont overfit or underfit data


# In[42]:


#Predictions on the dataset

# y_pred = gnb.predict([[2,0,1,0]])
y_pred = gnb.predict(X_test)


# In[43]:


y_pred


# In[47]:


from sklearn  import metrics 
print('Gaussian Naives Bayes Model accuracy: ' , metrics.accuracy_score(Y_test , y_pred) , "%") 


# ## RANDOM FORST CLASSIFIER

# ## Wine datasets from Sklearn.datasets

# In[58]:


from sklearn.datasets import load_wine


# In[66]:


wine_data = load_wine


# In[ ]:





# In[ ]:




