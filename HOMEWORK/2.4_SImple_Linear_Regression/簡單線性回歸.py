# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 23:05:07 2019

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os  #工作路徑用

os.chdir("D:\Machine Learning A-Z Chinese Template Folder\Part 2 - Regression\Section 4 - Simple Linear Regression");
#自動跳到工作路徑


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values   
Y = dataset.iloc[:, 1].values

#遺失數據處理
#from sklearn.impute import SimpleImputer
#imputer = SimpleImputer(missing_values =np.nan, strategy= 'mean')
#imputer.fit(X[:, 1:3])
#X[:, 1:3] = imputer.transform(X[:, 1:3])

#Splitting the datset into the Training set and Test set
from sklearn.model_selection import train_test_split  #自動將數據及分為訓練集及測試集
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 1/3, random_state = 0 )


#Feature Scaling,最常用的就是標準化
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)  #fit 會找出平均值與標準差並標準化
#X_test = sc.transform(X_test) #沒有fit 會直接用上面有fit過得到的平均值與標準差進行標準化


#Fiting Simple Linear Regression ti the Traning Set
from sklearn.linear_model import LinearRegression
re = LinearRegression()
re.fit(X_train, Y_train)

#Predicting the Test set results 預測測試集的因變量
y_pred = re.predict(X_test)


# Visualising the Training set results將數據視覺化
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, re.predict(X_train), color = 'blue')
plt.title('Salary VS Experience (training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()   #需與前面plt一同執行才會顯示這次

# Visualising the test set results將數據視覺化
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, re.predict(X_train), color = 'blue')
plt.title('Salary VS Experience (test set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()   #需與前面plt一同執行才會顯示這次

 


