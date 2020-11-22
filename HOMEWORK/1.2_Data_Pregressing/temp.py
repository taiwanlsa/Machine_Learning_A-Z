# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values     
#iloc代表會取這數據中的某些行列   :代表行數或列數
Y = dataset.iloc[:, 3].values

#遺失數據處理
from sklearn.impute import SimpleImputer
#用來處理遺失數據的
imputer = SimpleImputer(missing_values =np.nan, strategy= 'mean')
#可以用平均、中位數、眾數等等去填入遺失數據
imputer.fit(X[:, 1:3])
#此函數僅支援數值，所以必須給定列數，不然第0列會錯誤(地點為字串)
X[:, 1:3] = imputer.transform(X[:, 1:3])



#11.Encoding categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()   #new一個LableEncoder  
X[:,0] = labelencoder.fit_transform(X[:,0])  #將X第0列自動整理成數值資料並分類
#上述方法的分類會將無本質上差異的屬性變成有大小關係的數值，因此不好
#所以用oneHotEncoder的程式、以虛擬變量代替文字類別進行分析
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

Y = labelencoder.fit_transform(Y)


#Splitting the datset into the Training set and Test set
from sklearn.model_selection import train_test_split  #自動將數據及分為訓練集及測試集
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0 )
#以上會自動將X及Y分類

#Feature Scaling,最常用的就是標準化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  #fit 會找出平均值與標準差並標準化
X_test = sc.transform(X_test) #沒有fit 會直接用上面有fit過得到的平均值與標準差進行標準化








