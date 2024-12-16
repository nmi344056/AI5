'''
image 폴더꺼 수치화 (2만개)와 kaggle 폴더꺼 수치화 (2.5만개)를 합친다.
증폭 5천개 추가 (이미지 총 5만개)
만들고 kaggle에 제출
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.layers import Conv1D, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
np_path = 'C:\\ai5\\_data\\_save_npy\\keras43\\'
x_train1 = np.load(np_path + 'keras43_01_80_x_train.npy')
y_train1 = np.load(np_path + 'keras43_01_80_y_train.npy')
xy_test = np.load(np_path + 'keras43_01_80_x_test.npy')

path_np = 'C:\\ai5\\_data\\_save_npy\\keras45\\'
x_train2 = np.load(path_np + 'keras45_09_80_x_train.npy')
y_train2 = np.load(path_np + 'keras45_09_80_y_train.npy')

path = 'C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\'
mission = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(x_train1.shape, y_train1.shape)   # (25000, 80, 80, 3) (25000,)
print(x_train2.shape, y_train2.shape)   # (19997, 80, 80, 3) (19997,)
print(xy_test.shape)                    # (12500, 80, 80, 3)

x_train = np.concatenate((x_train1, x_train2))
y_train = np.concatenate((y_train1, y_train2))
print(x_train.shape, y_train.shape)     # (44997, 80, 80, 3) (44997,)

train_datagen = ImageDataGenerator(
    # rescale=1./255,           # 처음부터 0~1 사이의 스케일링한 데이터를 달라
    horizontal_flip=True,       # 수평 뒤집기
    vertical_flip=True,         # 수직 뒤집기
    width_shift_range=0.2,      # 평행 이동, 10%만큼 옆으로 이동
    # height_shift_range=0.1,   # 평행 이동 수직
    rotation_range=15,          # 정해진 각도만큼 이미지 회전
    # zoom_range=1.2,           # 축소 또는 확대
    # shear_range=0.7,          # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (일그러짐)
    fill_mode='nearest',        # 주변의 이미지의 비율대로 채운다
)

augment_size = 5003

randidx = np.random.randint(x_train.shape[0], size=augment_size)
# print(randidx)
# print(np.min(randidx), np.max(randidx))
print(x_train[0].shape)         # (80, 80, 3)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape, y_augmented.shape) # (5003, 80, 80, 3) (5003,)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]
print(x_augmented.shape)    # (5003, 80, 80, 3)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)     # (50000, 80, 80, 3) (50000,)

print(np.unique(y_train, return_counts=True))
# (array([0., 1.], dtype=float32), array([25005, 24995], dtype=int64))

x_train = x_train.reshape(50000, 80, 80*3)
xy_test = xy_test.reshape(12500, 80, 80*3)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8, random_state=123)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, input_shape=(80, 80*3), activation='relu'))
model.add(Conv1D(64, 2))
model.add(Dropout(0.2))
# model.add(Conv2D(32, 2, input_shape=(80, 80, 3), padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))

date = date.strftime("%m%d.%H%M")
print(date)
print(type(date))

path = './_save/keras59/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k59_05_catdog_date_', date, '_epo_', filename])

######### mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start2 = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[es, mcp])
end2 = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1, batch_size=16)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test, batch_size=16)
# print(y_predict)              # float 형
print(y_predict)                # batch_size가 너무 작으면 과적합으로 같은 값이 나올 수 있다.
# [[0.46824765]
#  [0.47400045]
#  [0.5960832 ]
#  ...
#  [0.46064526]
#  [0.6005819 ]
#  [0.56417537]]
print(y_test)                   # [0. 0. 1. ... 0. 1. 0.]
print(y_predict.shape)          # (10000, 1)
print(y_test.shape)             # (10000,)

y_predict = np.round(y_predict)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc)
# print("time :", round(end2 - start2, 2),'초')

y_submit = model.predict(xy_test, batch_size=1)
print(y_submit)
print(y_submit.shape)    # (12500, 1)

########## submission.csv 만들기 (count컬럼에 값만 넣으면 된다) ##########
mission['label'] = y_submit
print(mission)           # [12500 rows x 1 columns]
print(mission.shape)     # (12500, 1)

mission.to_csv(path + "submission_0806_21.csv")

'''
submission_0804_2248 > loss :  0.6086946725845337 / accuracy :  0.66 > 0.56463

augment
submission_0806_2053 > loss :  0.42448851466178894 / accuracy :  0.795 > 0.31537

ImageDataGenerator 수정
submission_0807_1100 > loss :  0.18178249895572662 / accuracy :  0.929 > 0.28989

LSTM
loss :  0.5828562378883362 / accuracy :  0.704

Conv1D
loss :  0.6928502321243286
accuracy :  0.5
(과적합)

'''
