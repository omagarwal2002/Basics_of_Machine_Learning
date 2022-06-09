# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 00:09:41 2022

@author: Dell
"""

from sklearn.datasets import make_regression

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# Regression data

X, y = make_regression(100, 5, shuffle=True, random_state=1, noise=50, bias=0.5)

x_train, x_test, y_train, y_test=  train_test_split(X, y , test_size=0.33)

# Regression model

reg = LinearRegression()

#reg.fit(X, y)

reg.fit(x_train, y_train)
#y_pred= reg.predict(X)

y_pred_train= reg.predict(x_train)

#score = r2_score(y, y_pred)
score_train = r2_score(y_pred_train, y_train)
print(score_train)

y_pred_test= reg.predict(x_test)
score_test = r2_score(y_pred_test, y_test)
print (score_test)

#%%

Noise= [0,10,50,70,100]
R2_Score_Train=[1.0,0.9809489156862062,0.6960758430828848,0.32594443800718165,0.22855480118561455]
R2_Score_Test=[1.0,0.9855359688851274,0.19202337086977395,0.3648884266019661,-0.2546747267351166]

plt.plot(Noise,R2_Score_Train, color ='red')
plt.plot(Noise,R2_Score_Test, color= 'green')
plt.show()

