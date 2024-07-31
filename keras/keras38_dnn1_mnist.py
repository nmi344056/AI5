# 35-4 카피

import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)  뒤에 1은 생략, reshape해서 (60000, 28, 28, 1)로 변경
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

########## 스케일링 1-1 ##########
x_train = x_train/255.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train))     # 1.0 0.0

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
print(x_train.shape, x_test.shape)      # (60000, 784) (10000, 784)

########## OneHotEncoding 1 ##########
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)      # (60000, 10) (10000, 10)

#2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(28*28,)))   # (26, 26, 64)
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)))   # (26, 26, 64)
# model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))        # (24, 24, 64)
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2), activation='relu'))                            # (23, 23, 32)
# model.add(Flatten())                                    # 23*23*32

# model.add(Dense(units=32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units=16, input_shape=(32,), activation='relu'))
# model.add(Dense(10, activation='softmax'))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))

date = date.strftime("%m%d.%H%M")
print(date)
print(type(date))

path = './_save/keras38/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k38_dnn1_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
# print(y_predict)            # float 형
# print(y_predict.shape)      # (10000, 10)

y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
print(y_predict)            #  int 형
print(y_predict.shape)      # (10000, 1)

y_test = np.argmax(y_test, axis=1).reshape(-1,1)
print(y_test)
print(y_test.shape)

acc = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc)
print("time :", round(end-start,2),'초')

print("++++++++++++++++++++")
print("loss : ", loss[0], "/ accuracy : ", round(loss[1], 3))

'''
[실습] accuracy 0.98 이상
cnn
loss :  0.02593575417995453 / accuracy :  0.9926 / time : 258.77 초

dnn
loss :  0.10085169970989227 / accuracy :  0.972 / time : 205.77 초

'''
