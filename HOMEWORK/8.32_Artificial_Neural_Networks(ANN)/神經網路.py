#Part1 data preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os  #工作路徑用

os.chdir("D:\Machine Learning A-Z Chinese Template Folder\Part 8 - Deep Learning\Section 32 - Artificial Neural Networks (ANN)");
#自動跳到工作路徑

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])  #將國家改成沒有大小之分的數據
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]  
#前三個變數，其實只需要兩個，因為第三個變數跟其他兩個變數互斥，例如法國=0 德國 = 0 西班牙必然=1
#會重複影響到分析品質，這邊選擇刪掉第一個



 
#Splitting the datset into the Training set and Test set
from sklearn.model_selection import train_test_split  #自動將數據及分為訓練集及測試集
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0 )


#Feature Scaling,最常用的就是標準化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  #fit 會找出平均值與標準差並標準化
X_test = sc.transform(X_test) #沒有fit 會直接用上面有fit過得到的平均值與標準差進行標準化


#part 2 make the ANN

#Importing the Keras librabaies and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN

classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform' , activation = 'relu' , input_dim = 11))
#隱藏層個數，這邊取輸入層的個數與輸出層的個數平均值
#Adding the secon hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform' , activation = 'relu' ))
#relu線性整合函數
#Adding the output layer
classifier.add(Dense(units = 1 , kernel_initializer = 'uniform' , activation = 'sigmoid' ))

#Compiling the ANN   編譯
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics = ['accuracy'] )
#大於三個結果的分析必須用categorical_crossentropy

#Fitting the ANN to the Training 
classifier.fit(X_train , Y_train, batch_size = 10 , epochs = 100)


#Part 3 - Making the predictions and evaluating the model



#Prediction the Test set results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred >0.5)


#MAking the confusion Matrix    評估多少預測正確用
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
