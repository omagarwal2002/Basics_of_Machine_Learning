# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 23:26:11 2022

@author: Dell
"""

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import linear_model

X, y = make_regression(100, 5, shuffle=True, random_state=4, noise=100, bias=0.5)
x_train, x_test, y_train, y_test=  train_test_split(X, y , test_size=0.33)
Noise= [0,30,50,70,100]
#%%
#linear regression
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(x_train, y_train)
y_pred_train= lin_regressor.predict(x_train)
score_train = r2_score(y_pred_train, y_train)
print(score_train)
y_pred_test= lin_regressor.predict(x_test)
score_test = r2_score(y_pred_test, y_test)
print (score_test)

#%%
R2_Score_Train=[1.0,0.9338632188959995,0.7681307987876425,0.5638818102318661,0.19341579821414578]
R2_Score_Test=[1.0,0.8780416755714645,0.807709667337179,0.5547090201137526,0.2435815002631775]

plt.plot(Noise,R2_Score_Train, color ='red')
plt.plot(Noise,R2_Score_Test, color= 'green')
plt.show()
#%%
#Ridge Regression
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge_regressor = linear_model.Lasso(alpha=0.1)
ridge_regressor.fit(x_train,y_train)
y_pred_train= ridge_regressor.predict(x_train)
score_train = r2_score(y_pred_train, y_train)
print(score_train)
y_pred_test= ridge_regressor.predict(x_test)
score_test = r2_score(y_pred_test, y_test)
print (score_test)

#%%
R2_Score_Train=[0.999995024924167,0.8966109961110564,0.7709951023565671,0.6109128967502577,0.10285665366728136]
R2_Score_Test=[0.9999960410078167,0.9299234763118096,0.7625033153069063,0.4257289171127968,-0.3506965355648457]

plt.plot(Noise,R2_Score_Train, color ='red')
plt.plot(Noise,R2_Score_Test, color= 'green')
plt.show()
#%%
#Lasso Regression
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso_regressor = linear_model.Lasso(alpha=0.1)
lasso_regressor.fit(x_train,y_train)
y_pred_train= lasso_regressor.predict(x_train)
score_train = r2_score(y_pred_train, y_train)
print(score_train)
y_pred_test= lasso_regressor.predict(x_test)
score_test = r2_score(y_pred_test, y_test)
print (score_test)

#%%
R2_Score_Train=[0.999993912857805,0.9215327005213542,0.7641070653813703,0.5207553752959468,0.18465374871908924]
R2_Score_Test=[0.999992817349997,0.9141181129090268,0.7464025562414421,0.5136900695122311,-0.2208005962922286]

plt.plot(Noise,R2_Score_Train, color ='red')
plt.plot(Noise,R2_Score_Test, color= 'green')
plt.show()
#%%
#Elastic Net Regression
from sklearn.linear_model import ElasticNet
elastic= ElasticNet()
elastic_regressor = linear_model.ElasticNet()
elastic_regressor.fit(x_train,y_train)
y_pred_train= elastic_regressor.predict(x_train)
score_train = r2_score(y_pred_train, y_train)
print(score_train)
y_pred_test= elastic_regressor.predict(x_test)
score_test = r2_score(y_pred_test, y_test)
print (score_test)

#%%
R2_Score_Train=[0.6816953388621252,0.4908848531758495,0.2002958355669644,-0.05786389200116937,-0.7035044333362195]
R2_Score_Test=[0.6590656871439343,0.18431401702049677,-0.3858669940297983,-0.05786389200116937,-2.0147826468034955]

plt.plot(Noise,R2_Score_Train, color ='red')
plt.plot(Noise,R2_Score_Test, color= 'green')
plt.show()