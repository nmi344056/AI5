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

y1 = np.array(range(3001, 3101)) # 수성의 온도
y2 = np.array(range(13001, 13101)) # 주식 가격

x4 = np.array([range(100, 105), range(401, 406)]).T
x5 = np.array([range(201, 206), range(511, 516), range(250, 255)]).T
x6 = np.array([range(100, 105), range(401, 406), range(177, 182), range(133, 138)]).T

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, \
        y1_train, y1_test, y2_train, y2_test = train_test_split(
        x1_datasets, x2_datasets, x3_datasets, y1, y2, train_size=0.8, random_state=3434)       # \는 줄바꿈

print(x1_train.shape, x1_test.shape)    # (80, 2) (20, 2)
print(x2_train.shape, x2_test.shape)    # (80, 3) (20, 3)
print(x3_train.shape, x3_test.shape)    # (80, 4) (20, 4)
print(y1_train.shape, y1_test.shape)      # (80,) (20,)
print(y2_train.shape, y2_test.shape)      # (80,) (20,)

#[실습] 분기하는 모델 만들기

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

#2-3. 모델 구성3
input12 = Input(shape=(4,))
dense12 = Dense(60, activation='relu', name='bit12')(input12)
dense22 = Dense(70, activation='relu', name='bit22')(dense12)
dense32 = Dense(80, activation='relu', name='bit32')(dense22)
output12 = Dense(90, activation='relu', name='bit42')(dense32)

#2-4. 모델 결합
# merge1 = concatenate([output1, output11], name='mg1')   # 2개 이상은 리스트
merge1 = Concatenate(name='mg1')([output1, output11, output12])     # concatenate과 동일
merge2 = Dense(7, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
middel_output = Dense(1, name='last')(merge3)
# model = Model(inputs=[input1, input11, input111], outputs=middel_output)

'''
전사영님이 알려준 코드, 모델 결합 및 분기
merge1 = Concatenate(name='mg1')([output1, output11, output111])
merge2 = Dense(7, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
last_output1 = Dense(1, name='last1')(merge3)
last_output2 = Dense(1, name='last2')(merge3)
model = Model(inputs=[input1, input11, input111], outputs=[last_output1, last_output2])
'''

#2-5. 모델 분기1
dense51 = Dense(15, activation='relu', name='bit51')(middel_output)
dense52 = Dense(25, activation='relu', name='bit52')(dense51)
dense53 = Dense(35, activation='relu', name='bit53')(dense52)
output51 = Dense(1, activation='relu', name='output51')(dense53)

#2-6. 모델 분기2
dense61 = Dense(16, activation='relu', name='bit61')(middel_output)
dense62 = Dense(26, activation='relu', name='bit62')(dense61)
output61 = Dense(1, activation='relu', name='output61')(dense62)

model = Model(inputs=[input1, input11, input12], outputs=[output51, output61])

model.summary()
'''
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_1 (InputLayer)           [(None, 2)]          0           []
 bit1 (Dense)                   (None, 10)           30          ['input_1[0][0]']
 input_3 (InputLayer)           [(None, 4)]          0           []
 bit2 (Dense)                   (None, 20)           220         ['bit1[0][0]']
 input_2 (InputLayer)           [(None, 3)]          0           []
 bit12 (Dense)                  (None, 60)           300         ['input_3[0][0]']
 bit3 (Dense)                   (None, 30)           630         ['bit2[0][0]']
 bit11 (Dense)                  (None, 100)          400         ['input_2[0][0]']
 bit22 (Dense)                  (None, 70)           4270        ['bit12[0][0]']
 bit4 (Dense)                   (None, 40)           1240        ['bit3[0][0]']
 bit21 (Dense)                  (None, 200)          20200       ['bit11[0][0]']
 bit32 (Dense)                  (None, 80)           5680        ['bit22[0][0]']
 bit5 (Dense)                   (None, 50)           2050        ['bit4[0][0]']
 bit31 (Dense)                  (None, 300)          60300       ['bit21[0][0]']
 bit42 (Dense)                  (None, 90)           7290        ['bit32[0][0]']
 mg1 (Concatenate)              (None, 440)          0           ['bit5[0][0]',
                                                                  'bit31[0][0]',
                                                                  'bit42[0][0]']
 mg2 (Dense)                    (None, 7)            3087        ['mg1[0][0]']
 mg3 (Dense)                    (None, 20)           160         ['mg2[0][0]']
 last (Dense)                   (None, 1)            21          ['mg3[0][0]']
 bit51 (Dense)                  (None, 15)           30          ['last[0][0]']
 bit52 (Dense)                  (None, 25)           400         ['bit51[0][0]']
 bit61 (Dense)                  (None, 16)           32          ['last[0][0]']
 bit53 (Dense)                  (None, 35)           910         ['bit52[0][0]']
 bit62 (Dense)                  (None, 26)           442         ['bit61[0][0]']
 output51 (Dense)               (None, 1)            36          ['bit53[0][0]']
 output61 (Dense)               (None, 1)            27          ['bit62[0][0]']
==================================================================================================
Total params: 107,755
Trainable params: 107,755
Non-trainable params: 0
'''

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
filepath = "".join([path, 'k62_03_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=1000, batch_size=1, validation_split=0.2, verbose=3, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print("loss :", loss)

y_predict = model.predict([x4, x5, x6])
# print(y_predict.shape)  # (1, 144)

print('loss :',loss, '(',round(end-start,2),'초)')
print('예측값 [3101, 3102, 2103, 3104, 3105], [13101, 13102, 12103, 13104, 13105] : ', y_predict[0], y_predict[1])

'''
loss : [1363611.125, 62752.86328125, 1300858.25] ( 3.61 초)
예측값 [3101, 3102, 2103, 3104, 3105], [13101, 13102, 12103, 13104, 13105] :  [array([[3367.3987],
       [3376.0012],
       [3384.6033],
       [3393.2058],
       [3401.8076]], dtype=float32), array([[14914.099],
       [14952.201],
       [14990.301],
       [15028.401],
       [15066.502]], dtype=float32)]

'''
