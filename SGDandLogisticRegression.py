# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 22:26:49 2022

@author: Dell
"""
#%%
#import numpy as  np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
classes = 2
X,t= make_classification(100, 5, n_classes = classes, random_state= 40, n_informative = 2, n_clusters_per_class = 1)
#%%
X_train, X_test, y_train, y_test=  train_test_split(X, t , test_size=0.50)
#%%
model2 = SGDClassifier(loss='log')
model = LogisticRegression(penalty = 'elasticnet', solver= 'saga', verbose= 1, class_weight=None, multi_class='auto' ,l1_ratio=0.2)
#%%
model.fit(X_train, y_train)
model2.fit(X_train, y_train)
#%%
y=model.predict(X_test)
y2=model.predict(X_train)
y3=model2.predict(X_test)
#%%
print(model.coef_)
print(model.intercept_)
print(model2.coef_)
print(model2.intercept_)
#%%
from sklearn.metrics import accuracy_score
score2 =accuracy_score(y, y_test)
score =accuracy_score(y2, y_train)
print(score2)
print(score)
score3 =accuracy_score(y3, y_test)
print(score3)
