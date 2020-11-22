#設定工作路徑
setwd("D:/Machine Learning A-Z Chinese Template Folder/Part 1 - Data Preprocessing/Section 2 - Part 1 - Data Preprocessing -/Data_Preprocessing")
#windows得把斜線都改成右斜才能用


# Importing the dataset
dataset = read.csv('Data.csv')
#dataset = dataset[, 2:3]

#Splitting the dataset into the Training set and Test set
library(caTools)  #直接導入
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature Scaling
#training_set[,2:3] = scale(training_set[,2:3])  
#test_set[,2:3]  = scale(test_set[,2:3])