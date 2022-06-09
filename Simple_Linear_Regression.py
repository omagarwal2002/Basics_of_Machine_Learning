# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 12:03:37 2022

@author: Dell
"""
#f(x) = L / 1 + e^-k(x - x0)
#y = e^(b0 + b1*x) / (1 + e^(b0 + b1*x))
#%%
import numpy as np

from sklearn.datasets import make_regression 

X , t = make_regression (100,1, shuffle = True , bias = 0.0 ,noise = 40, random_state= 4) ;

#%%
import matplotlib.pyplot as plt
plt.scatter(X,t)

#%%
mean_x = np.mean(np.squeeze(X)) 
mean_t = np.mean(t)
std_x  = np.std(X)
std_t = np.std(t)

#%%
d_x= X- mean_x
d_t= t- mean_t

num = np.sum(np.squeeze(d_x)*d_t)

deno = np.sum(np.squeeze(d_x)*np.squeeze(d_x))

B1= num/ deno 
B0= mean_t - B1*mean_x

print(B1)
print(B0)

#%%
y= B0 + B1*X[50,0]
print(y)
print(t[50])

