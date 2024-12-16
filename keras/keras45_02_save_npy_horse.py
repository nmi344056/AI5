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
path_train = './_data/image/horse_human/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100,100),  # resize, 동일한 규격 사용
    batch_size=2000,
    class_mode='binary',
    color_mode='rgb',
)   # Found 1027 images belonging to 2 classes.

np_path = 'C:\\ai5\\_data\\_save_npy\\keras45\\'
np.save(np_path + 'keras45_02_horse_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras45_02_horse_y_train.npy', arr=xy_train[0][1])
