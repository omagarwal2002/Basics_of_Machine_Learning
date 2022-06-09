# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 21:21:28 2022

@author: Dell
"""

#%%
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
classes = 3
X,t= make_classification(100, 5, n_classes = classes, random_state= 40, n_informative = 2, n_clusters_per_class = 1)

#%%
res = np.zeros((t.shape[0], classes), dtype=int)
res[np.arange(t.shape[0]), t] = 1
x0=np.ones((100,1), dtype=int);
new_x = np.concatenate((x0,X), axis = 1);
train_X, test_X, train_t, test_t= train_test_split(new_x ,res, shuffle = True )
tr=train_X.transpose();
m3 = np. dot(tr,train_X);
i = np.linalg.inv(m3);
m4 = np. dot(i,tr);
m5=np.dot(m4,train_X);
transp=m5.transpose();

#%%
print("the value of W is:-");
print(m5);
y = np. dot(test_X,m5);
print("the value of y is", y );
k=np.argmax(y , axis = 1);
print("the value of k is:- ",k);

count=0
for i in k:
    if(i==1 or i==2):
        count+=1
accuracy=(count/len(k))*100
print("the accuracy is:-", accuracy)
