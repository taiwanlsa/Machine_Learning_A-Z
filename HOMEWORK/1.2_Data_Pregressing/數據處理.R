# Importing the dataset
dataset = read.csv('Data.csv')

#Taking care of missing data
dataset$Age[is.na(dataset$Age)] = mean(dataset$Age, na.rm = T)
  #dataset數據集中Age的欄位表示dataset$Age
  #is.na()函數提供R快速檢驗資料集中是否含有遺漏值
  #na.rm  na代表nun rm代表remove
dataset$Salary[is.na(dataset$Salary)] = mean(dataset$Salary, na.rm = T)



#Encoding categorical data
dataset$Country =  factor(dataset$Country, 
                          levels = c('France', 'Spain', 'Germany'), 
                          labels = c(1,2,3))  
#R語言中此方法會指定為為分類型態，但python沒有，並且分類型態不可運算
#分類型態沒有大小之分，僅為區別分類辨識用
dataset$Purchased =  factor(dataset$Purchased, 
                          levels = c('No', 'Yes'), 
                          labels = c(0,1))


#Splitting the datset into the Training set and Test set
# install.packages('caTools')
library(caTools)  #直接導入
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
#要加入$Purchased，才會讓電腦知道要這行的全部數據分割，
#否則會認為是四個類別，ration代表隨機分配比例
#R語言中不需要new一個funtion來用

training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#以下為自行測試用
#X = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
#X1 = subset(X, split == TRUE)
#X2 = subset(dataset, split == FALSE)



#Feature Scaling
training_set[,2:3] = scale(training_set[,2:3])  
#因國家及是否購買兩列為分類因子(分類型態)不可運算，因此指定23列標準化
test_set[,2:3]  = scale(test_set[,2:3])










