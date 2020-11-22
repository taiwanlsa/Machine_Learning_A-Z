#Polynomial Regression 回歸

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os  #工作路徑用多項式

os.chdir("D:\Machine Learning A-Z Chinese Template Folder\Part 2 - Regression\Section 6 - Polynomial Regression");
#自動跳到工作路徑

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values   #為什麼要變成矩陣？？？？
Y = dataset.iloc[:, 2].values

#遺失數據處理
#from sklearn.impute import SimpleImputer
#imputer = SimpleImputer(missing_values =np.nan, strategy= 'mean')
#imputer.fit(X[:, 1:3])
#X[:, 1:3] = imputer.transform(X[:, 1:3])

#Splitting the datset into the Training set and Test set
#from sklearn.model_selection import train_test_split  #自動將數據及分為訓練集及測試集
#_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0 )

#Feature Scaling,最常用的就是標準化
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)  #fit 會找出平均值與標準差並標準化
#X_test = sc.transform(X_test) #沒有fit 會直接用上面有fit過得到的平均值與標準差進行標準化


#Fitting Linear Refression ti the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Fitting Polynomial Regression to the sataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)


#Visualising the Linear Regression results
plt.scatter(X,Y,color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylebel('Salary')
plt.show()

#Visualising the Polynomial Regression results
X_grid = np.arange(min(X),max(X) , 0.1) #將原本的X數據分割為0.1為單位
X_grid = X_grid.reshape(len(X_grid), 1) #轉置矩陣  以上兩行只是為了讓圖形看起來比較平滑
plt.scatter(X,Y,color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylebel('Salary')
plt.show()

# Predicing a new result with Linear Regression
lin_reg.predict(6.5)

# Predicing a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))






