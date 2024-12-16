# [복사] keras07_2_1

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
             [9,8,7,6,5,4,3,2,1,0],])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x = x.T

print(x.shape)          # (10, 3)
print(y.shape)          # (10,)

#2-1. 순차형 모델 구성
model = Sequential()
model.add(Dense(10, input_shape=(3,)))
model.add(Dense(9))
model.add(Dropout(0.3))
model.add(Dense(8))
model.add(Dropout(0.2))
model.add(Dense(7))
model.add(Dense(1))

model.summary()
'''
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 10)                40         input=3
 dense_1 (Dense)             (None, 9)                 99
 dense_2 (Dense)             (None, 8)                 80
 dense_3 (Dense)             (None, 7)                 63
 dense_4 (Dense)             (None, 1)                 8
=================================================================
Total params: 290
Trainable params: 290
Non-trainable params: 0
'''

#2-2. 함수형 모델 구성
# input1 = Input(shape=(3,))
# dense1 = Dense(10)(input1)
# dense2 = Dense(9)(dense1)
# dense3 = Dense(8)(dense2)
# dense4 = Dense(7)(dense3)
# output1 = Dense(1)(dense4)
# model = Model(inputs=input1, outputs=output1)

# model.summary()
'''
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 3)]               0
 dense_5 (Dense)             (None, 10)                40
 dense_6 (Dense)             (None, 9)                 99
 dense_7 (Dense)             (None, 8)                 80
 dense_8 (Dense)             (None, 7)                 63
 dense_9 (Dense)             (None, 1)                 8
=================================================================
Total params: 290
Trainable params: 290
Non-trainable params: 0
'''

# input1 = Input(shape=(3,))
# dense1 = Dense(10, name='ys1')(input1)
# dense2 = Dense(9, name='ys2')(dense1)
# dense3 = Dense(8, name='ys3')(dense2)
# dense4 = Dense(7, name='ys4')(dense3)
# output1 = Dense(1)(dense4)
# model = Model(inputs=input1, outputs=output1)

# model.summary()
'''
 Layer (type)                Output Shape              Param #
=================================================================
 input_2 (InputLayer)        [(None, 3)]               0
 ys1 (Dense)                 (None, 10)                40
 ys2 (Dense)                 (None, 9)                 99
 ys3 (Dense)                 (None, 8)                 80
 ys4 (Dense)                 (None, 7)                 63
 dense_10 (Dense)            (None, 1)                 8
=================================================================
Total params: 290
Trainable params: 290
Non-trainable params: 0
'''

input1 = Input(shape=(3,))
dense1 = Dense(10, name='ys1')(input1)
dense2 = Dense(9, name='ys2')(dense1)
drop1 = Dropout(0.3)(dense2)
dense3 = Dense(8, name='ys3')(drop1)
drop2 = Dropout(0.2)(dense3)
dense4 = Dense(7, name='ys4')(drop2)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)

model.summary()
