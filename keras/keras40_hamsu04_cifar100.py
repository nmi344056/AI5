import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1) 컬러 데이터
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

# print(np.unique(y_train, return_counts=True))

x_train = x_train/255.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) # 1.0 0.0

# x_train = x_train.reshape(50000, 32*32*3)
# x_test = x_test.reshape(10000, 32*32*3)
# print(x_train.shape, x_test.shape)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
print(y_train.shape, y_test.shape)  # (50000, 100) (10000, 100)

#2. 모델 구성
input1 = Input(shape=(32, 32, 3,))
dense1 = Conv2D(64, (3,3), activation='relu', strides=1, padding='same')(input1)
Pooling1 = MaxPooling2D()(dense1)
drop1 = Dropout(0.3)(Pooling1)
dense2 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', strides=1, padding='same')(drop1)
Pooling2 = MaxPooling2D()(dense2)
drop2 = Dropout(0.2)(Pooling2)
dense3 = Conv2D(32, (2,2), activation='relu', strides=1, padding='same')(drop2)
Pooling2 = MaxPooling2D()(dense3)
Flatten1 = Flatten()(Pooling2)

dense4 = Dense(32, activation='relu')(Flatten1)
drop2 = Dropout(0.3)(dense4)
dense5 = Dense(units=16, input_shape=(32,), activation='relu')(drop2)
output1 = Dense(100, activation='softmax')(dense5)
model = Model(inputs=input1, outputs=output1)

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

path = './_save/keras35/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k35_cnn7_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1, batch_size=256, validation_split=0.2, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
# print(y_predict)            # float 형
# print(y_predict.shape)      # (10000, 100)

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
cnn
loss :  3.3071742057800293 / accuracy :  0.188 / time : 159.44 초

dnn
loss :  3.4617156982421875 / accuracy :  0.185 / time : 40.12 초

'''
