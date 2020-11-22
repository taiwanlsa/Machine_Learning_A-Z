import os  #工作路徑用
os.chdir("D:\Machine Learning A-Z Chinese Template Folder\Part 8 - Deep Learning\Section 33 - Convolutional Neural Networks (CNN)");
#自動跳到工作路徑


#Part1 data preprocessing  已經用資料夾分類好了


#Importing the Keras librabaries and packages

from keras.models import Sequential     #初始化神經網路
from keras.layers import Convolution2D  #創建一個捲積層
from keras.layers import MaxPooling2D   #在神經網路內添加池化層
from keras.layers import Flatten        #扁平層
from keras.layers import Dense          #添加全連接層

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution

classifier.add(Convolution2D(32,(3,3), activation = 'relu', input_shape =(64,64,3) ))
#特徵探測器數目()32就夠了，過大會影響效能，(3,3)3*3的特徵探測器大小
#input_shape =(64,64,3)代表圖片大小64*64，並且3色

#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Part 3 -改善模型準確度，Addinig a second convolutional layer
classifier.add(Convolution2D(32,(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))



#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))  #隱藏層
#128是測試來的
classifier.add(Dense(units = 1, activation = 'sigmoid'))  #輸出層

#Compiling the CNN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])



#Part 2 - Fitting the CNN th the image
from keras.preprocessing.image import ImageDataGenerator

#直接使用官方說明書的範例改編  
#網址https://keras-cn.readthedocs.io/en/latest/legacy/preprocessing/image/
train_datagen = ImageDataGenerator(    #圖片生成器
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),   #想要圖片的大小
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#計算時間用
import time
t1 = time.time()

#會跑超級久
classifier.fit_generator(
        training_set,
        steps_per_epoch=250,   
        epochs = 25,
        validation_data = test_set,
        validation_steps =62.5)
#samples_per_epoch=250,bechsize是32，有8000張圖片，所以8000/32

t2 = time.time()
print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')

