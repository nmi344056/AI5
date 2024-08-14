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
x3_datasets = np.array([range(100), range(301, 401), range(77, 177), range(33, 133)]).T

y = np.array(range(3001, 3101)) # 수성의 온도

x4 = np.array([range(100, 105), range(401, 406)]).T
x5 = np.array([range(201, 206), range(511, 516), range(250, 255)]).T
x6 = np.array([range(100, 105), range(401, 406), range(177, 182), range(133, 138)]).T

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(x1_datasets, x2_datasets, x3_datasets, y, train_size=0.8, random_state=3434)

print(x1_train.shape, x1_test.shape)    # (80, 2) (20, 2)
print(x2_train.shape, x2_test.shape)    # (80, 3) (20, 3)
print(x3_train.shape, x3_test.shape)    # (80, 4) (20, 4)
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

#2-2. 모델 구성2
input11 = Input(shape=(3,))
dense11 = Dense(100, activation='relu', name='bit11')(input11)
dense21 = Dense(200, activation='relu', name='bit21')(dense11)
output11 = Dense(300, activation='relu', name='bit31')(dense21)
# model2 = Model(inputs=input11, outputs=output11)
# model2.summary()

#2-2. 모델 구성3
input111 = Input(shape=(4,))
dense111 = Dense(60, activation='relu', name='bit111')(input111)
dense211 = Dense(70, activation='relu', name='bit211')(dense111)
dense311 = Dense(80, activation='relu', name='bit311')(dense211)
output111 = Dense(90, activation='relu', name='bit411')(dense311)

#2-4. 모델 결합
# merge1 = concatenate([output1, output11], name='mg1')   # 2개 이상은 리스트
merge1 = Concatenate(name='mg1')([output1, output11, output111])     # concatenate과 동일
merge2 = Dense(7, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)
model = Model(inputs=[input1, input11, input111], outputs=last_output)

model.summary()
'''
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_1 (InputLayer)           [(None, 2)]          0           []
 bit1 (Dense)                   (None, 10)           30          ['input_1[0][0]']
 input_3 (InputLayer)           [(None, 4)]          0           []
 bit2 (Dense)                   (None, 20)           220         ['bit1[0][0]']
 input_2 (InputLayer)           [(None, 3)]          0           []
 bit111 (Dense)                 (None, 60)           300         ['input_3[0][0]']
 bit3 (Dense)                   (None, 30)           630         ['bit2[0][0]']
 bit11 (Dense)                  (None, 100)          400         ['input_2[0][0]']
 bit211 (Dense)                 (None, 70)           4270        ['bit111[0][0]']
 bit4 (Dense)                   (None, 40)           1240        ['bit3[0][0]']
 bit21 (Dense)                  (None, 200)          20200       ['bit11[0][0]']
 bit311 (Dense)                 (None, 80)           5680        ['bit211[0][0]']
 bit5 (Dense)                   (None, 50)           2050        ['bit4[0][0]']
 bit31 (Dense)                  (None, 300)          60300       ['bit21[0][0]']
 bit411 (Dense)                 (None, 90)           7290        ['bit311[0][0]']
 mg1 (Concatenate)              (None, 440)          0           ['bit5[0][0]',
                                                                  'bit31[0][0]',
                                                                  'bit411[0][0]']
 mg2 (Dense)                    (None, 7)            3087        ['mg1[0][0]']
 mg3 (Dense)                    (None, 20)           160         ['mg2[0][0]']
 last (Dense)                   (None, 1)            21          ['mg3[0][0]']
==================================================================================================
Total params: 105,878
Trainable params: 105,878
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
filepath = "".join([path, 'k62_02_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
model.fit([x1_train, x2_train, x3_train], y_train, epochs=1000, batch_size=1024, validation_split=0.2, verbose=3, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y_test)
print("loss :", loss)

y_predict = model.predict([x4, x5, x6])
# print(y_predict.shape)  # (1, 144)

print('loss :',loss, '(',round(end-start,2),'초)')
print('예측값 [3101, 3102, 2103, 3104, 3105] : ', y_predict)

'''
 loss : 0.06690039485692978 ( 28.76 초) > k62_02_date_0814.1431_epo_0746_valloss_0.0099
예측값 [3101, 3102, 2103, 3104, 3105] :  [[3103.109 ]
 [3106.6643]
 [3110.2373]
 [3113.768 ]
 [3117.338 ]]
 
 loss : 2.007124423980713 ( 20.26 초) > k62_02_date_0814.1433_epo_0435_valloss_0.6660
예측값 [3101, 3102, 2103, 3104, 3105] :  [[3102.4653]
 [3105.2573]
 [3108.3596]
 [3111.6284]
 [3115.2588]]
 
'''
