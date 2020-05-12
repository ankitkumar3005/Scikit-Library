#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset= pd.read_csv('Iris.csv')


# In[3]:


dataset


# In[17]:


x = dataset.iloc[:,1:5].values


# In[19]:


dataset.iloc[:,1:5].values


# In[16]:


x


# In[6]:


y = dataset.iloc[:,5].values


# In[7]:


dataset.iloc[:,5]


# In[10]:


from sklearn import datasets


# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[12]:


labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# In[13]:


y


# In[20]:


x


# In[21]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size = 0.2)


# In[23]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)


# In[24]:


y_pred = logmodel.predict(x_test)


# In[25]:


y_pred


# In[26]:


y_test


# In[29]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[30]:


28/30


# In[35]:


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5 , metric = 'minkowski' , p=2)
classifier_knn.fit(x_train,y_train)


# In[38]:


y_pred = classifier_knn.predict(x_test)


# In[39]:


confusion_matrix(y_test,y_pred)


# In[40]:


28/30 #accuracy


# In[42]:


from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(x_train,y_train)


# In[43]:


y_pred = classifier_nb.predict(x_test)


# In[46]:


confusion_matrix(y_test,y_pred)


# In[45]:


29/30


# In[48]:


from sklearn.svm import SVC
classifier_svm_sigmoid = SVC(kernel = 'sigmoid')
classifier_svm_sigmoid.fit(x_train,y_train)


# In[52]:


y_pred = classifier_svm_sigmoid.predict(x_test)


# In[53]:


confusion_matrix(y_test,y_pred)


# In[55]:


9/30


# In[56]:


from sklearn.svm import SVC
classifier_svm_linear = SVC(kernel = 'linear')
classifier_svm_linear.fit(x_train,y_train)


# In[57]:


y_pred = classifier_svm_linear.predict(x_test)


# In[58]:


confusion_matrix(y_test,y_pred)


# In[59]:


28/30


# In[60]:


from sklearn.svm import SVC
classifier_svm_rbf = SVC(kernel = 'rbf')
classifier_svm_rbf.fit(x_train,y_train)


# In[62]:


y_test = classifier_svm_rbf.predict(x_test)


# In[64]:


confusion_matrix(y_test,y_pred)


# In[65]:


30/30


# In[66]:


from sklearn.svm import SVC
classifier_svm_poly = SVC(kernel = 'poly')
classifier_svm_poly.fit(x_train,y_train)


# In[67]:


y_test = classifier_svm_poly.predict(x_test)


# In[68]:


confusion_matrix(y_test,y_pred)


# In[69]:


30/30


# In[77]:


from sklearn.tree import DecisionTreeClassifier
classifier_dt  = DecisionTreeClassifier(criterion = 'entropy')
classifier_dt.fit(x_train,y_train)


# In[78]:


y_pred = classifier_dt.predict(x_test)


# In[79]:


confusion_matrix(y_test,y_pred)


# In[76]:


30/30


# In[80]:


from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 3, criterion = 'entropy')


# In[81]:


classifier_rf.fit(x_train,y_train)


# In[82]:


y_pred = classifier_rf.predict(x_test)


# In[83]:


confusion_matrix(y_test,y_pred)

