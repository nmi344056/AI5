import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#1. 데이터
x_datasets = np.array([range(100), range(301, 401)]).transpose()

#[실습] 일대다 모델 만들기

y1 = np.array(range(3001, 3101)) # 수성의 온도
y2 = np.array(range(13001, 13101)) # 주식 가격

x_predict = np.array([range(100, 105), range(401, 406)]).T

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
        x_datasets, y1, y2, train_size=0.8, random_state=123)       # \는 줄바꿈

print(x1_train.shape, x1_test.shape)    # (80, 2) (20, 2)
print(y1_train.shape, y1_test.shape)      # (80,) (20,)
print(y2_train.shape, y2_test.shape)      # (80,) (20,)

#2-1. 모델 구성1
input1 = Input(shape=(2,))
dense1 = Dense(16, activation='relu', name='bit1')(input1)
dense2 = Dense(32, activation='relu', name='bit2')(dense1)
dense3 = Dense(64, activation='relu', name='bit3')(dense2)
dense4 = Dense(32, activation='relu', name='bit4')(dense3)
output1 = Dense(16, activation='relu', name='bit5')(dense4)

#2-5. 모델 분기1
# dense51 = Dense(15, activation='relu', name='bit51')(middel_output)
# dense52 = Dense(25, activation='relu', name='bit52')(dense51)
# dense53 = Dense(35, activation='relu', name='bit53')(dense52)
output51 = Dense(1, activation='relu', name='output51')(output1)

#2-6. 모델 분기2
# dense61 = Dense(16, activation='relu', name='bit61')(middel_output)
# dense62 = Dense(26, activation='relu', name='bit62')(dense61)
output61 = Dense(1, activation='relu', name='output61')(output1)

model = Model(inputs=input1, outputs=[output51, output61])

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))

date = date.strftime("%m%d.%H%M")
print(date)
print(type(date))

path = './_save/keras62/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k62_04_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
model.fit(x1_train, [y1_train, y2_train], epochs=1000, batch_size=2, validation_split=0.2, verbose=3, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x1_test, [y1_test, y2_test])
print("loss :", loss)

y_predict = model.predict(x_predict)
# print(y_predict.shape)  # (1, 144)

print('loss :',loss, '(',round(end-start,2),'초)')
print('예측값 [3101, 3102, 2103, 3104, 3105], [13101, 13102, 12103, 13104, 13105] : ', y_predict)

'''
loss : [4.056096258864272e-06, 3.197789283149177e-06, 8.583068620282575e-07] ( 102.15 초) > k62_04_date_0815.2239_epo_0833_valloss_0.0000
예측값 [3101, 3102, 2103, 3104, 3105], [13101, 13102, 12103, 13104, 13105] :  [array([[3100.9988],
       [3101.9988],
       [3102.9993],
       [3103.9993],
       [3104.9993]], dtype=float32), array([[13101.001],
       [13102.   ],
       [13103.001],
       [13104.   ],
       [13105.001]], dtype=float32)]

train_size=0.2로 되어있어서 loss가 컸다. 항상 기초를 잊지 말자.

'''
