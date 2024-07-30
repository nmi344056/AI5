import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)  뒤에 1은 생략, reshape해서 (60000, 28, 28, 1)로 변경
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape, x_test.shape)      # (60000, 28, 28, 1) (10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)      # (60000, 10) (10000, 10)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(28, 28, 1)))   # (27, 27, 10)
                            # shape = (batch_size, rows, columns, channels)
                            # shape = (batch_size, heights, widths, channels)
                            #                     (       input_shape       )
model.add(Conv2D(filters=20, kernel_size=(3,3)))     # Conv2D(20, (3,3))과 동일, (25, 25, 20)
model.add(Conv2D(15, (4,4)))    # (22, 22, 15)
model.add(Flatten())

model.add(Dense(units=8))
model.add(Dense(units=9, input_shape=(8,)))
                            # shape = (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))

model.summary()
'''
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 27, 27, 10)        50
 conv2d_1 (Conv2D)           (None, 25, 25, 20)        1820
 conv2d_2 (Conv2D)           (None, 22, 22, 15)        4815
 dense (Dense)               (None, 22, 22, 8)         128
 dense_1 (Dense)             (None, 22, 22, 9)         81
=================================================================
Total params: 6,894
Trainable params: 6,894
Non-trainable params: 0

model.add(Flatten()) 추가
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 27, 27, 10)        50
 conv2d_1 (Conv2D)           (None, 25, 25, 20)        1820
 conv2d_2 (Conv2D)           (None, 22, 22, 15)        4815
 flatten (Flatten)           (None, 7260)              0
 dense (Dense)               (None, 8)                 58088
 dense_1 (Dense)             (None, 9)                 81
=================================================================
Total params: 64,854
Trainable params: 64,854
Non-trainable params: 0
'''

# #3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True,)

# ########## mcp 세이브 파일명 만들기 시작 ##########
# import datetime
# date = datetime.datetime.now()
# print(date)
# print(type(date))

# date = date.strftime("%m%d.%H%M")
# print(date)
# print(type(date))

# path = './_save/keras35_mcp/'
# filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
# filepath = "".join([path, 'k35_cnn3_date_', date, '_epo_', filename])

# ########## mcp 세이브 파일명 만들기 끝 ##########

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

# start = time.time()
# hist = model.fit(x_train, y_train, epochs=2, batch_size=128, validation_split=0.2, callbacks=[es, mcp])
# end = time.time()

# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test, verbose=1)
# print("loss : ", loss[0])
# print("accuracy : ", round(loss[1], 3))

# y_pre = model.predict(x_test)
# r2 = r2_score(y_test,y_pre)
# print('r2 score :', r2)
# print("time :", round(end-start,2),'초')

# '''
# CPU
# loss :  0.5863330364227295
# accuracy :  0.85
# r2 score : 0.7442206420929631
# time : 23.41 초

# GPU
# loss :  0.5339111089706421
# accuracy :  0.855
# r2 score : 0.7561774905134075
# time : 4.78 초
# '''
