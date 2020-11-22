#Multiple Linear Regression
#設定工作路徑
setwd("D:/Machine Learning A-Z Chinese Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression")
#windows得把斜線都改成右斜才能用


# Importing the dataset
dataset = read.csv('50_Startups.csv')

dataset$State = factor(dataset$State,
                         levels = c('New York', 'California','Florida'),
                         labels = c(1,2,3))



#Splitting the datset into the Training set and Test set
library(caTools)  #直接導入
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature Scaling
#training_set[,2:3] = scale(training_set[,2:3])  
#test_set[,2:3]  = scale(test_set[,2:3])


#擬合線性回歸
#regressor = lm(formula = Profit ~R.D.Spend + Administration +Marketing.Spend +State,
#               data = training_set)

regressor = lm(formula = Profit ~.,
               data = training_set)   #用一個.代表除了Profit其他的都是因變量

#predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

#building the optima; using Backward Elimination 反覆淘汰法，淘汰不顯著的因變量
regressor = lm(formula = Profit ~R.D.Spend + Marketing.Spend,
               data = training_set)
summary(regressor)


