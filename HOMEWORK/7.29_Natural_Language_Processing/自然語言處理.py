#
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os  #工作路徑用

os.chdir("D:\Machine Learning A-Z Chinese Template Folder\Part 7 - Natural Language Processing\Section 29 - Natural Language Processing");
#自動跳到工作路徑

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t' , quoting = 3)  
#quoting 要去除掉


#Cleaning the texts 
import re
import nltk
nltk.download('stopwords')   #已經建立好的虛詞字典
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(1000):
    
    
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])   #清除非字母的符號
    review = review.lower()   #將所有大寫轉乘小寫
    review = review.split()
    ps = PorterStemmer()
    
    #review = [word for word in review if not word in stopwords.words('english')]
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #比上一行多了一個set可以加快速度 WTF?
    #ps.stem() 此函數用意為將單詞取詞根化，像是過去式、分詞等等還原成原來的單詞
    
    review = ' '.join(review)
    
    corpus.append(review) 
#Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 200) #只針對最常出現的1500個單詞做分析
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:,1].values
 


#Splitting the datset into the Training set and Test set
from sklearn.model_selection import train_test_split  #自動將數據及分為訓練集及測試集
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 0 )


# fitting Naive Bayes the Training set
#creat your classifier here
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)



#Prediction the Test set results
Y_pred = classifier.predict(X_test)

#MAking the confusion Matrix    評估多少預測正確用
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
 
