# 65_04 copy

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Embedding
from tensorflow.keras.layers import Conv1D, Conv2D, Flatten,  Input, Dropout
from tensorflow.keras.layers import Concatenate, concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없었어요', '너무 재미없다', '참 재밋네요',
    '준영이 바보', '반장 잘생겼다', '태운이 또 구라친다',
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화예요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별
# 로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없었어요': 21, '재미없다': 22, '재밋네요': 23, '준영이': 24, '바보': 25, '반장': 26, '잘생겼다': 27, '태운이': 28, '또': 29, '구라친다': 30}
# 1부터 시작 주의

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6], 
# [7, 8, 9], [10, 11, 12, 13, 14], [15], 
# [16], [17, 18], [19, 20], 
# [21], [2, 22], [1, 23], 
# [24, 25], [26, 27], [28, 29, 30]]
print(type(x))      # <class 'list'>

from tensorflow.keras.preprocessing.sequence import pad_sequences
x1 = pad_sequences(x)  # Default : padding='pre', value=0
print(x1, x1.shape)
'''
[[ 0  0  0  2  3]
 [ 0  0  0  1  4]
 [ 0  0  1  5  6]
 [ 0  0  7  8  9]
 [10 11 12 13 14]
 [ 0  0  0  0 15]
 [ 0  0  0  0 16]
 [ 0  0  0 17 18]
 [ 0  0  0 19 20]
 [ 0  0  0  0 21]
 [ 0  0  0  2 22]
 [ 0  0  0  1 23]
 [ 0  0  0 24 25]
 [ 0  0  0 26 27]
 [ 0  0 28 29 30]] (15, 5)
'''

x_pre = ['태운이 참 재미없다.'] # 리스트 형태로

# token = Tokenizer()         # x와 같은 token을 사용하기 위해 해당 세 줄은 주석
# token.fit_on_texts(x_pre)
# print(token.word_index)     # {'태운이': 1, '참': 2, '재미없다': 3}

# x_pre = '태운이 참 재미없다.'
# x_pre = token.texts_to_sequences([x_pre])
x_pre = token.texts_to_sequences(x_pre)
print(x_pre)                # [[28, 1, 22]]
# print(x_pre.shape)        # 에러 (1, 3)

x_pre = pad_sequences(x_pre, 5)
print(x_pre, x_pre.shape)   # [[ 0  0 28  1 22]] (1, 5)

x_train, x_test, y_train, y_test = train_test_split(x1, labels, train_size=0.8, random_state=123)

#2. 모델 구성
model = Sequential()

##### Embedding 1 #####
# model.add(Embedding(input_dim=31, output_dim=100, input_length=5))
# model.add(LSTM(10))
# model.add(Dense(10))
# model.add(Dense(1, activation='sigmoid'))
'''
 Layer (type)                Output Shape              Param #
=================================================================
 embedding (Embedding)       (None, 5, 100)            3100
 lstm (LSTM)                 (None, 10)                4440
 dense (Dense)               (None, 10)                110
 dense_1 (Dense)             (None, 1)                 11
=================================================================
Total params: 7,661
Trainable params: 7,661
Non-trainable params: 0
'''

##### Embedding 2 #####
# model.add(Embedding(input_dim=31, output_dim=100))
# model.add(LSTM(10))
# model.add(Dense(10))
# model.add(Dense(1, activation='sigmoid'))
'''
 Layer (type)                Output Shape              Param #
=================================================================
 embedding (Embedding)       (None, None, 100)         3100
 lstm (LSTM)                 (None, 10)                4440
 dense (Dense)               (None, 10)                110
 dense_1 (Dense)             (None, 1)                 11
=================================================================
Total params: 7,661
Trainable params: 7,661
Non-trainable params: 0
'''

##### Embedding 3 #####
# model.add(Embedding(input_dim=100, output_dim=100))   # 아무런 숫자나 넣어도 작동한다, 성능 차이
# model.add(LSTM(10))
# model.add(Dense(10))
# model.add(Dense(1, activation='sigmoid'))

# input_dim=30    # 디폴트
# input_dim=20    # 단어사전의 갯수보다 작을 때 : 연산량 줄어듬, 단어사전에서 임으로 뺀다 : 성능 조금 저하
# input_dim=40    # 단어사전의 갯수보다 클 때 : 연산량 늘어남, 임의의 랜덤 임베딩 생성 : 성능 조금 저하

##### Embedding 4 #####
# model.add(Embedding(31, 100))                     # 정상 작동
# model.add(Embedding(31, 100, 5))                  # ValueError: Could not interpret initializer identifier: 5
# model.add(Embedding(31, 100, input_length=5))     # 정상 작동
# model.add(Embedding(31, 100, input_length=6))     # ValueError: Input 0 of layer "sequential" is incompatible with the layer: expected shape=(None, 6), found shape=(1, 5)
model.add(Embedding(31, 100, input_length=1))       # 작동, 약수(1, 5)는 가능, WARNING:tensorflow:Model was constructed with shape (None, 1) for input KerasTensor(type_spec=TensorSpec(shape=(None, 1), dtype=tf.float32, name='embedding_input'), name='embedding_input', description="created by layer 'embedding_input'"), but it was called on an input with incompatible shape (None, 5).
model.add(LSTM(10))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))

date = date.strftime("%m%d.%H%M")
print(date)
print(type(date))

path = './_save/keras65/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k65_07_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es, mcp])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_pre)
print(y_predict)              # float 형
# print(y_predict.shape)      # (10000, 10)

y_predict = np.round(y_predict)
print("태운이 참 재미없다. : ", y_predict)

'''
loss :  0.6806584000587463
accuracy :  0.667
[[0.48948905]]
태운이 참 재미없다. :  [[0.]]

loss :  0.6831741333007812
accuracy :  1.0
[[0.49796057]]
태운이 참 재미없다. :  [[0.]]

'''
