# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:35:41 2022

@author: Dell
"""

#%%
from sklearn.datasets import make_regression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#%%
X_reg,t_reg = make_regression(100, 3, n_informative=15, noise=0.1, random_state=6)
X_train,X_test,t_train,t_test=train_test_split(X_reg,t_reg,test_size=0.3,random_state=2)

#%%
model_reg = AdaBoostRegressor()
model_reg.fit(X_train, t_train)
#%%
y_pred_train = model_reg.predict(X_train)

score_train = r2_score(y_pred_train, t_train)
print(score_train)
#%%
y_pred_test = model_reg.predict(X_test)
score_test = r2_score(y_pred_test, t_test)
print(score_test)
#%%
from sklearn import tree

tree.plot_tree(model_reg.estimators_[0], filled=True)
