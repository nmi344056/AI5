# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset

import numpy as np
import pandas as pd
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

#1. 데이터
start1 = time.time()
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # rotation_range=1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'      
)

test_datagen = ImageDataGenerator(rescale=1./255,)

path_train = "C:/ai5/_data/kaggle/biggest-gender/faces"

xy_train = train_datagen.flow_from_directory(
    path_train,            
    target_size=(100,100),  
    batch_size=30000,          
    class_mode='binary',  
    color_mode='rgb',  
    shuffle=True, 
)   # Found 27167 images belonging to 2 classes.

# print(xy_train[0][0])           # 
# print(xy_train[0][0].shape)     # (27167, 100, 100, 3)
# print(xy_train[0][1])           # [0. 0. 0. ... 0. 0. 1.]
# print(xy_train[0][1].shape)     # (27167,)

path_np = 'C:\\ai5\\_data\\_save_npy\\keras45\\'
np.save(path_np + 'keras45_07_gender_x_train.npy', arr=xy_train[0][0])
np.save(path_np + 'keras45_07_gender_y_train.npy', arr=xy_train[0][1])
