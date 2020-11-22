
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os  #工作路徑用

os.chdir("D:\Machine Learning A-Z Chinese Template Folder\Part 5 - Association Rule Learning\Section 22 - Apriori");
#自動跳到工作路徑

#importing the data
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)  #沒有標題
transations = []
for i in range(0,7501):
    transations.append([str(dataset.values[i,j]) for j in range(0,20)])


#以下為錯誤用法會變成每個list僅有一個字串
transations2 = []
for i in range(0,7501):
    for j in range(0,20):
        transations2.append(str(dataset.values[i,j]))
        
        
#training Apriort on the dataset
import apyori as ap
rules = ap.apriori(transations, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2) 
#需要花時間確認MIN_confidence需要用多少

#visualising the results
results = list(rules)   #相關度
myresults = pd.DataFrame([list(x) for x in results])  #叫出商品名稱




