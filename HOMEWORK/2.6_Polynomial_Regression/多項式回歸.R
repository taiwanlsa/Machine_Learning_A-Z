setwd("D:/Machine Learning A-Z Chinese Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression")


# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

# library(caTools)  #直接導入
# set.seed(123)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3]  = scale(test_set[,2:3])

#Simple Regression
Lin_reg = lm(formula = Salary~.,
             data = dataset)

dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
dataset$Level5 = dataset$Level^5
dataset$Level6 = dataset$Level^6
dataset$Level7 = dataset$Level^7
poly_reg = lm(formula = Salary~Level5+Level6+Level7,
              data = dataset)

#visualising the Linear Regression results
install.packages('ggplot2')
library(ggplot2)
ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             color = 'red') +
  geom_line(aes(x=dataset$Level, y = predict(Lin_reg, nemdata = dataset)),
            color = 'blue') +
  xlab('Level') + 
  ylab('Salary')


ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             color = 'red') +
  geom_line(aes(x=dataset$Level, y = predict(poly_reg, nemdata = dataset)),
            color = 'blue') +
  xlab('Level') + 
  ylab('Salary')


#predicting a new result with Linear Regression
y_pred = predict(Lin_reg,data.frame(Level = 6.5))   #新增一個數來預測


#predicting a new result with Polynomial Regression
y_pred = predict(poly_reg,data.frame(Level = 6.5^1 ,Level2 = 6.5^2,
                                     Level3 = 6.5^3 ,Level4 =6.5 ^4, 
                                     Level5 =6.5^5,Level6 = 6.5^6, Level7 =6.5^7))   

