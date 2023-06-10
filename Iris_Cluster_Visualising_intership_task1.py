#!/usr/bin/env python
# coding: utf-8

# <h1>Sparks Foundation Internship Task-2</h1>

# <h3>Problem Statement Predict the optimal number of clusters and represent it visually</h3>

# Importing the required modules

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Reading the file

# In[2]:


Iris = pd.read_csv('Iris.csv')
Iris.head()


# In[3]:


Iris.info()


# In[ ]:


#Checking to null values


# In[4]:


Iris.isnull().sum()


# In[22]:


Iris.describe()


# In[6]:


Iris.shape


# In[ ]:


#checking the correlation of the data with each other


# In[23]:


sns.pairplot(Iris)


# Imporing kmeans for further prediction

# In[41]:


#importing Kmeans
from sklearn.cluster import KMeans
x = Iris.iloc[:, [0, 1, 2, 3,4]].values #taking all the rows and columns until petalwidthcm
wcss =[]  #Within-Cluster Sum of Square 
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# In[42]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[43]:


y_kmeans


# In[44]:


kmeans_pred = kmeans.fit_predict(x)
kmeans_pred


# In[45]:


Iris['Cluster']=kmeans_pred


# In[46]:


fig = plt.figure(1, figsize=(5, 6))
Iris['Cluster'].value_counts().plot(kind='bar')
plt.legend()


# In[47]:


fig = plt.figure(1, figsize=(11, 7))
plt.scatter(x[kmeans_pred == 0, 0], x[kmeans_pred == 0, 1], alpha=0.7, label = 'Iris-setosa', color='yellow')
plt.scatter(x[kmeans_pred == 1, 0], x[kmeans_pred == 1, 1], alpha=0.7, label = 'Iris-versicolour', color='green')
plt.scatter(x[kmeans_pred == 2, 0], x[kmeans_pred == 2, 1], alpha=0.7, label = 'Iris-virginica', color='violet')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, marker='*', c='black', label = 'Centroids')

plt.legend()


# In[ ]:




