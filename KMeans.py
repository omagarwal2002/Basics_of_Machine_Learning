# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 11:24:38 2022

@author: Dell
"""

from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
#%%
classes = 2
X, t= make_classification(100, 5, n_classes = classes, random_state= 40, n_clusters_per_class= 2)
#%%
X_train, X_test, y_train, y_test=  train_test_split(X,t, test_size=0.33)
#%%
model = KMeans(n_clusters=5, random_state=0)
model.fit(X_train)
#%%
y1=model.predict(X_train)
y=model.predict(X_test)
print(y)
print(y1)
#%%
from sklearn import metrics 
score = metrics.accuracy_score(y_test,y)
print(score)
#%%
#visualizing clusters with kmeans
import matplotlib.pyplot as plt
plt.scatter(X[:,0], 
            X[:,1])
plt.scatter(model.cluster_centers_[:, 0], 
            model.cluster_centers_[:, 1], 
            s=200,                             # Set centroid size
            c='red')                           # Set centroid color
plt.show()