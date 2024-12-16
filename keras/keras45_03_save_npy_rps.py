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

train_datagen = ImageDataGenerator(rescale=1./255,)
path_train = './_data/image/rps/'

start1 = time.time()

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100,100),  # resize, 동일한 규격 사용
    batch_size=3000,
    # class_mode='binary',          # 이진분류
    # class_mode='categorical',       # 다중분류 - onehot도 되서 나온다
    class_mode='sparse',          # 다중분류 - onehot 하기 이전 상태 (분류하기 전)
    # class_mode='None',            # y값 생략 (summit은 이걸로)
    # color_mode='grayscale',
    color_mode='rgb',
    shuffle=True,
)   # Found 19997 images belonging to 2 classes.

np_path = 'C:\\ai5\\_data\\_save_npy\\keras45\\'
np.save(np_path + 'keras45_03_rps_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras45_03_rps_y_train.npy', arr=xy_train[0][1])

end1 = time.time()
print("time :", round(end1 - start1, 2),'초')    # time : 46.8 초

print(np.unique(xy_train, return_counts=True))