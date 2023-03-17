#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import plotly.express as ex

import seaborn as sns


# In[2]:


data = pd.read_csv("spam.csv", encoding= 'latin-1')
data = data[["class", "message"]]


# In[3]:


# data.head()


# In[4]:


# print(data.groupby('class').size())


# In[5]:


sns.countplot(data['class'])


# In[6]:


# data.loc[data["class" ]== "spam"]


# In[8]:


## Preparing data for modeling


# In[9]:


x = np.array(data["message"])
y = np.array(data["class"])


# In[10]:


cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data


# In[11]:


# Training the Model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[12]:


## Creating the Model


# In[13]:


clf = MultinomialNB()
clf.fit(X_train,y_train)


# In[14]:


## Tesing the Model


# In[15]:


# sample = input('Enter a message:')
# data = cv.transform([sample]).toarray()
# print(clf.predict(data))


# In[16]:


## Deploying Model


# In[19]:


import streamlit as st
st.title("Spam Detection System")
def spamdetection():
    user = st.text_area("Enter any Message or Email: ")
    if len(user) < 1:
        st.write("Insufficient length")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        a = clf.predict(data)
        st.title(a)
spamdetection()

