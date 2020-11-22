# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 22:37:24 2019

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os  #工作路徑用

os.chdir("D:\Machine Learning A-Z Chinese Template Folder\Part 2 - Regression\Section 5 - Multiple Linear Regression");
#自動跳到工作路徑

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values   
Y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()   #new一個LableEncoder  
X[:,3] = labelencoder.fit_transform(X[:,3])  #將X第3列自動整理成數值資料並分類
#上述方法的分類會將無本質上差異的屬性變成有大小關係的數值，因此不好
#所以用oneHotEncoder的程式、以虛擬變量代替文字類別進行分析
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X[:, 1:]  #取所有數據，但只取第一項之後的類別




#Splitting the datset into the Training set and Test set
from sklearn.model_selection import train_test_split  #自動將數據及分為訓練集及測試集
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0 )


#Feature Scaling,最常用的就是標準化
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)  #fit 會找出平均值與標準差並標準化
#X_test = sc.transform(X_test) #沒有fit 會直接用上面有fit過得到的平均值與標準差進行標準化


#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
re = LinearRegression()
re.fit(X_train,Y_train)  #就是求一個多元線性回歸解

#Predicting the Test set results
y_pred = re.predict(X_test) #用測試集的數據求個預測值



#Building the optimal model using Backward Elination
#以p-value來觀察哪些自變量沒有太大影響，以及那些因變量不需要
import statsmodels.formula.api as sm
#一個方程式應該包含一個常數項，上方的回歸並不會給常數項，因此要手動去弄
X_train = np.append(arr = np.ones((40,1)),values = X_train, axis = 1)    #加上第一列的1
X_opt = X_train[: , [0,1,2,3,4,5]]  #optimal最佳  
re_OLS = sm.OLS(endog = Y_train , exog = X_opt).fit()
re_OLS.summary()


X_opt = X_train[: , [0,1,3,4,5]]  #optimal最佳  
re_OLS = sm.OLS(endog = Y_train , exog = X_opt).fit()
re_OLS.summary()

X_opt = X_train[: , [0,3,4,5]]  #optimal最佳  
re_OLS = sm.OLS(endog = Y_train , exog = X_opt).fit()
re_OLS.summary()

X_opt = X_train[: , [0,3,5]]  #optimal最佳  
re_OLS = sm.OLS(endog = Y_train , exog = X_opt).fit()
re_OLS.summary()

X_opt = X_train[: , [0,3]]  #optimal最佳  
re_OLS = sm.OLS(endog = Y_train , exog = X_opt).fit()
re_OLS.summary()





