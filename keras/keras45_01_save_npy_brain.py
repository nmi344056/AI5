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
from sklearn.metrics import r2_score, accuracy_score

train_datagen = ImageDataGenerator(rescale=1./255,)
test_datagen = ImageDataGenerator(rescale=1./255,)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(200,200),  # resize, 동일한 규격 사용
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)   # Found 160 images belonging to 2 classes.  (ad, normal 합 총 160개)

xy_test = test_datagen.flow_from_directory(
    path_train,
    target_size=(200,200),  # resize, 동일한 규격 사용
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale',
)   # Found 160 images belonging to 2 classes.

path_np = 'C:\\ai5\\_data\\_save_npy\\keras45\\'
np.save(path_np + 'keras45_01_brain_x_train.npy', arr=xy_train[0][0])
np.save(path_np + 'keras45_01_brain_y_train.npy', arr=xy_train[0][1])
np.save(path_np + 'keras45_01_brain_x_test.npy', arr=xy_test[0][0])
np.save(path_np + 'keras45_01_brain_y_test.npy', arr=xy_test[0][1])
