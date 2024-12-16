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
path_np = 'C:\\ai5\\_data\\_save_npy\\keras45\\'
x_train = np.load(path_np + 'keras45_07_gender_x_train.npy')
y_train = np.load(path_np + 'keras45_07_gender_y_train.npy')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=921)
end1 = time.time()

print('데이터 걸린시간 :',round(end1-start1,2),'초')
# 데이터 걸린시간 : 1.43 초

print(x_train.shape, y_train.shape) # (20000, 100, 100, 3) (20000,)
print(x_test.shape, y_test.shape)   # (5000, 100, 100, 3) (5000,)
# 데이터 걸린시간 : 48.61 초

#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.7))

model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))

date = date.strftime("%m%d.%H%M")
print(date)
print(type(date))

path = './_save/keras45/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k45_07_gender_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath, )

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.1, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1,
                      batch_size=32
                      )
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = model.predict(x_test,batch_size=16)
# r2 = r2_score(y_test,y_pre)
# print('r2 score :', r2)
print("걸린 시간 :", round(end-start,2),'초')

y_pre = np.round(y_pre)
r2 = accuracy_score(y_test, y_pre)
print('accuracy_score :', r2)

# 제출용 csv 파일은 없다.

'''
loss : 0.2086048424243927 / acc : 0.91866
'''
