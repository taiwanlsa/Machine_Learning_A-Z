#設定工作路徑
setwd("D:/Machine Learning A-Z Chinese Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression")
#windows得把斜線都改成右斜才能用

# Importing the dataset
dataset = read.csv('Salary_Data.csv')


#Splitting the dataset into the Training set and Test set
library(caTools)  #直接導入
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature Scaling
#training_set[,2:3] = scale(training_set[,2:3])  
#test_set[,2:3]  = scale(test_set[,2:3])


#Fitting Simple Linear Regression to the Training set
re = lm(formula = Salary  ~ YearsExperience,
        data = training_set)   #想要預測的是Salary

#Console指令區輸入：summary(re) 可得知回歸後的各項指標


#predicting the Test set results
y_pred = predict(re, newdata = test_set)

#Visualising the Training set result
#install.packages('ggplot2')  #安裝
library(ggplot2)
ggplot() +
  geom_point(aes(x= training_set$YearsExperience , y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x= training_set$YearsExperience , y = predict(re,newdata = training_set)),
                colour = 'blue') +
  ggtitle('薪水VS經驗') +
  xlab('經驗') +
  ylab('薪水')
  

#Visualising the test set result
library(ggplot2)
ggplot() +
  geom_point(aes(x= test_set$YearsExperience , y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x= training_set$YearsExperience , y = predict(re,newdata = training_set)),
            colour = 'blue') +
  ggtitle('薪水VS經驗') +
  xlab('經驗') +
  ylab('薪水')

  
  
            





