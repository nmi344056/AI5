import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#1. 데이터
x_datasets = np.array([range(100), range(301, 401)]).transpose()

#[실습] 일대다 모델 만들기

y1 = np.array(range(3001, 3101)) # 수성의 온도
y2 = np.array(range(13001, 13101)) # 주식 가격

x4 = np.array([range(100, 105), range(401, 406)]).T

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
        x_datasets, y1, y2, train_size=0.8, random_state=312)       # \는 줄바꿈

print(x1_train.shape, x1_test.shape)    # (80, 2) (20, 2)
print(y1_train.shape, y1_test.shape)      # (80,) (20,)
print(y2_train.shape, y2_test.shape)      # (80,) (20,)

# #2-1. 모델 구성1
# input1 = Input(shape=(2,))
# dense1 = Dense(16, activation='relu', name='bit1')(input1)
# dense2 = Dense(64, activation='relu', name='bit2')(dense1)
# dense3 = Dense(128, activation='relu', name='bit3')(dense2)
# dense4 = Dense(64, activation='relu', name='bit4')(dense3)
# dense5 = Dense(32, activation='relu', name='bit5')(dense4)
# output1 = Dense(1, activation='relu', name='output1')(dense5)
# output2 = Dense(1, activation='relu', name='output2')(dense5)
# model = Model(inputs=input1, outputs=[output1, output2])

# # model.summary()

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)

# ########## mcp 세이브 파일명 만들기 시작 ##########
# import datetime
# date = datetime.datetime.now()
# print(date)
# print(type(date))

# date = date.strftime("%m%d.%H%M")
# print(date)
# print(type(date))

# path = './_save/keras62/'
# filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
# filepath = "".join([path, 'k62_04_date_', date, '_epo_', filename])

# ########## mcp 세이브 파일명 만들기 끝 ##########

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

# start = time.time()
# model.fit(x1_train, [y1_train, y2_train], epochs=1000, batch_size=1024, validation_split=0.2, verbose=3, callbacks=[mcp])
# end = time.time()

#4. 평가, 예측
path2 = './_save/keras62/'
model = load_model(path2 + 'k62_04_date_0815.2239_epo_0833_valloss_0.0000.hdf5')

loss = model.evaluate(x1_test, [y1_test, y2_test])
print("loss :", loss)

y_predict = model.predict(x4)
# print(y_predict.shape)  # (1, 144)

# print('loss :',loss, '(',round(end-start,2),'초)')
print('예측값 [3101, 3102, 2103, 3104, 3105], [13101, 13102, 12103, 13104, 13105] : ', y_predict[0], y_predict[1])

'''
loss : [470.9013671875, 425.3418884277344, 45.559478759765625] ( 35.84 초)
예측값 [3101, 3102, 2103, 3104, 3105], [13101, 13102, 12103, 13104, 13105] :  [[3066.6138]
 [3066.9417]
 [3067.2688]
 [3067.5923]
 [3067.913 ]] [[13108.931]
 [13110.087]
 [13111.24 ]
 [13112.373]
 [13113.49 ]]





'''
