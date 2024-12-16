import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).transpose()
        # (2, 100) -> (100, 2) , jena 데이터라고 가정
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).T
        # (3, 100) -> (100, 3) , mnist 데이터라고 가정

y = np.array(range(3001, 3101)) # 수성의 온도

x3 = np.array([range(100, 105), range(401, 406)]).T
x4 = np.array([range(201, 206), range(511, 516), range(250, 255)]).T

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_datasets, x2_datasets, y, train_size=0.8, random_state=3434)

print(x1_train.shape, x1_test.shape)    # (80, 2) (20, 2)
print(x2_train.shape, x2_test.shape)    # (80, 3) (20, 3)
print(y_train.shape, y_test.shape)      # (80,) (20,)

#2-1. 모델 구성1
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='bit1')(input1)
dense2 = Dense(20, activation='relu', name='bit2')(dense1)
dense3 = Dense(30, activation='relu', name='bit3')(dense2)
dense4 = Dense(40, activation='relu', name='bit4')(dense3)
output1 = Dense(50, activation='relu', name='bit5')(dense4)
# model1 = Model(inputs=input1, outputs=output1)
# model1.summary()
'''
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 2)]               0
 bit1 (Dense)                (None, 10)                30
 bit2 (Dense)                (None, 20)                220
 bit3 (Dense)                (None, 30)                630
 bit4 (Dense)                (None, 40)                1240
 bit5 (Dense)                (None, 50)                2050
=================================================================
Total params: 4,170
Trainable params: 4,170
Non-trainable params: 0
'''

#2-2. 모델 구성2
input11 = Input(shape=(3,))
dense11 = Dense(100, activation='relu', name='bit11')(input11)
dense21 = Dense(200, activation='relu', name='bit21')(dense11)
output11 = Dense(300, activation='relu', name='bit31')(dense21)
# model2 = Model(inputs=input11, outputs=output11)
# model2.summary()
'''
 Layer (type)                Output Shape              Param #
=================================================================
 input_2 (InputLayer)        [(None, 3)]               0
 bit11 (Dense)               (None, 100)               400
 bit21 (Dense)               (None, 200)               20200
 bit31 (Dense)               (None, 300)               60300
=================================================================
Total params: 80,900
Trainable params: 80,900
Non-trainable params: 0
'''

#[검색] concatenate 모델 결합

#2-3. 모델 결합
# merge1 = concatenate([output1, output11], name='mg1')   # 2개 이상은 리스트
merge1 = Concatenate(name='mg1')([output1, output11])     # concatenate과 동일
merge2 = Dense(7, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)
model = Model(inputs=[input1, input11], outputs=last_output)

model.summary()
'''
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_1 (InputLayer)           [(None, 2)]          0           []
 bit1 (Dense)                   (None, 10)           30          ['input_1[0][0]']
 bit2 (Dense)                   (None, 20)           220         ['bit1[0][0]']
 input_2 (InputLayer)           [(None, 3)]          0           []
 bit3 (Dense)                   (None, 30)           630         ['bit2[0][0]']
 bit11 (Dense)                  (None, 100)          400         ['input_2[0][0]']
 bit4 (Dense)                   (None, 40)           1240        ['bit3[0][0]']
 bit21 (Dense)                  (None, 200)          20200       ['bit11[0][0]']
 bit5 (Dense)                   (None, 50)           2050        ['bit4[0][0]']
 bit31 (Dense)                  (None, 300)          60300       ['bit21[0][0]']
 mg1 (Concatenate)              (None, 350)          0           ['bit5[0][0]',
                                                                  'bit31[0][0]']
 mg2 (Dense)                    (None, 7)            2457        ['mg1[0][0]']
 mg3 (Dense)                    (None, 20)           160         ['mg2[0][0]']
 last (Dense)                   (None, 1)            21          ['mg3[0][0]']
==================================================================================================
Total params: 87,708
Trainable params: 87,708
Non-trainable params: 0
'''

#[실습] 만들기 (주의사항 : 2개 이상은 리스트)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)

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
filepath = "".join([path, 'k62_01_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
model.fit([x1_train, x2_train], y_train, epochs=1000, batch_size=1024, validation_split=0.2, verbose=3, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print("loss :", loss)

y_predict = model.predict([x3, x4])
# print(y_predict.shape)  # (1, 144)

print('loss :',loss, '(',round(end-start,2),'초)')
print('예측값 [3101, 3102, 2103, 3104, 3105] : ', y_predict)

'''
loss : 0.00019082128710579127 ( 36.22 초) > k62_jena_date_0814.1258_epo_0854_valloss_0.0002
예측값 :  [[3090.9827]
 [3066.9854]
 [3098.9822]
 [3092.983 ]
 [3035.9883]]
 
 -----
 loss : 0.20772497355937958 ( 30.76 초)
예측값 [3101, 3102, 2103, 3104, 3105] :  [[3103.8923]
 [3107.6934]
 [3111.622 ]
 [3115.8042]
 [3120.4412]]
 
 loss : 0.000676766037940979 ( 28.63 초)
예측값 [3101, 3102, 2103, 3104, 3105] :  [[3105.0261]
 [3109.1008]
 [3113.175 ]
 [3117.2498]
 [3121.328 ]]
 
'''
