#邏輯迴歸處理程式集

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os  #工作路徑用



os.chdir("D:\Machine Learning A-Z Chinese Template Folder\Part 4 - Clustering\Section 19 - K-Means Clustering");
#自動跳到工作路徑


#屬於無監督式學習，不須有因變量
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 3:5].values   

#Using the elbow method th find the optmal number of clusters

import sklearn.cluster as cl
wcss = []
for i in range(1,11):
    kmeans = cl.KMeans(n_clusters = i , max_iter = 300, n_init = 10, init = 'k-means++', random_state =123)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('Elbow Function')
plt.xlabel('number of Clusters')
plt.ylabel('WCSS')
plt.show

#applying the k-means to the mall dataset
kmeans = cl.KMeans(n_clusters = 5 , max_iter = 300, n_init = 10, init = 'k-means++', random_state =0)
Y_kmeans = kmeans.fit_predict(X)

#Visualizing the clusters

plt.scatter(X[Y_kmeans ==0,0],X[Y_kmeans == 0,1], s =100 ,c = 'red'   , label = 'Cluster 0-> carefu; ')
plt.scatter(X[Y_kmeans ==1,0],X[Y_kmeans == 1,1], s =100 ,c = 'blue'  , label = 'Cluster 1-> standard ')
plt.scatter(X[Y_kmeans ==2,0],X[Y_kmeans == 2,1], s =100 ,c = 'green' , label = 'Cluster 2-> target ')
plt.scatter(X[Y_kmeans ==3,0],X[Y_kmeans == 3,1], s =100 ,c = 'orange', label = 'Cluster 3-> Careless ')
plt.scatter(X[Y_kmeans ==4,0],X[Y_kmeans == 4,1], s =100 ,c = 'purple', label = 'Cluster 4-> Sensible ')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1],  s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show

