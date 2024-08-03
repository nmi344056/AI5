# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview

# 42_1 복사 (#1. 데이터만)

import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
#     horizontal_flip=True,
#     vertical_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=5,
#     zoom_range=1.2,
#     shear_range=0.7,
#     fill_mode='nearest',
)

test_datagen = ImageDataGenerator(rescale=1./255,)

path_train = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\train\\'
path_test2 = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\test2\\'
path = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\'

mission = pd.read_csv(path + "sample_submission.csv", index_col=0)
print(mission)      # [12500 rows x 1 columns]

start1 = time.time()

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(80,80),  # resize, 동일한 규격 사용
    batch_size=25000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)   # Found 25000 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test2,
    target_size=(80,80),  # resize, 동일한 규격 사용
    batch_size=12500,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False, 
)   # Found 

# print(xy_train[0][0].shape)

np_path = 'C:\\ai5\\_data\\_save_npy\\keras43\\'
np.save(np_path + 'keras43_01_80_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras43_01_80_y_train.npy', arr=xy_train[0][1])
np.save(np_path + 'keras43_01_80_x_test.npy', arr=xy_test[0][0])
np.save(np_path + 'keras43_01_80_y_test.npy', arr=xy_test[0][1])

# 넘파이 저장은 train_test_split 하기 전 - train_test_split의 비율, 랜덤을 바꾸기 위해

end1 = time.time()
print("time :", round(end1 - start1, 2),'초')    # time : 46.8 초
