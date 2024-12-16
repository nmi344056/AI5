# [실습] 15개의 행에서 5개를 더 넣어서 만들기 (2개 행은 6개 이상 단어, 자르기)

# [실습] x shape : (15, 5, 31)
# x_predict shape : (1, 5, 31)

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
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
    '또 보고 싶은 잘만든 영화예요 참 재밋네요', '연기가 별로에요',
    '너무 잘만든 영화예요', '얼굴이 재밋네요','인생 최악의 영화입니다 다들 꼭 보세요'
]

labels = np.array([1,1,1,
                   1,1,0,
                   0,0,0,
                   0,0,1,
                   0,1,0,
                   1,0,
                   1,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '잘만든': 3, '영화예요': 4, '재밋네요': 5, '싶은': 6, '영화입니다': 7, '보고': 8, '별로에요': 9, '연기가': 10, '또': 11, '재미있다': 12, '최고에요': 13, '추천하고': 14, '한': 15, '번': 16, '더': 17, '싶어요': 18, '글 
# 쎄': 19, '생각보다': 20, '지루해요': 21, '어색해요': 22, '재미없었어요': 23, '재미없다': 24, '준영이': 25, '바보': 26, '반장': 27, '잘생겼다': 28, '태운이': 29, '구라친다': 30, '얼굴이': 31, '인생': 32, '최악의': 33, '다들': 34, '꼭': 35, '보세요': 36}
# 1부터 시작 주의

x = token.texts_to_sequences(docs)
print(x)
# [[2, 12], [1, 13], [1, 3, 4], 
# [14, 6, 7], [15, 16, 17, 8, 18], [19], 
# [9], [20, 21], [10, 22], 
# [23], [2, 24], [1, 5], 
# [25, 26], [27, 28], [29, 11, 30], 
# [11, 8, 6, 3, 4, 1, 5], [10, 9], 
# [2, 3, 4], [31, 5], [32, 33, 7, 34, 35, 36]]
print(type(x))      # <class 'list'>

from tensorflow.keras.preprocessing.sequence import pad_sequences
x1 = pad_sequences(x, 5)  # Default : padding='pre', value=0
print(x1, x1.shape)
'''
[[ 0  0  0  2 12]
 [ 0  0  0  1 13]
 [ 0  0  1  3  4]
 [ 0  0 14  6  7]
 [15 16 17  8 18]
 [ 0  0  0  0 19]
 [ 0  0  0  0  9]
 [ 0  0  0 20 21]
 [ 0  0  0 10 22]
 [ 0  0  0  0 23]
 [ 0  0  0  2 24]
 [ 0  0  0  1  5]
 [ 0  0  0 25 26]
 [ 0  0  0 27 28]
 [ 0  0 29 11 30]
 [ 6  3  4  1  5]
 [ 0  0  2  3  4]
 [ 0  0  0 31  5]
 [33  7 34 35 36]] (20, 5)
'''

x_pre = '태운이 참 재미없다.' # 리스트 형태로
x_pre = token.texts_to_sequences([x_pre])
print(x_pre)                # [[29, 1, 24]]
# print(x_pre.shape)        # 에러 (1, 3)

x_pre = pad_sequences(x_pre, 5)
print(x_pre, x_pre.shape)   # [[ 0  0 28  1 22]] (1, 5)

from tensorflow.keras.utils import to_categorical
x_ohe = to_categorical(x1)
# xp_ohe = to_categorical(x_pre)    # num_classes을 지정하지 않으면 가장 큰 수인 28로 원핫 -> (1,5,29)로 나온다 (get_dummies 등은 채우지 않고 (1,5,4)로 나온다)
xp_ohe = to_categorical(x_pre, num_classes=37)
print(x_ohe, x_ohe.shape)           # (20, 5, 37)
print(xp_ohe, xp_ohe.shape)         # (1, 5, 37)

x_train, x_test, y_train, y_test = train_test_split(x_ohe, labels, train_size=0.8, random_state=123)

#2. 모델 구성
model = Sequential()
model.add(LSTM(16, input_shape=(5, 37), return_sequences=True))
model.add(LSTM(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)

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
filepath = "".join([path, 'k65_06_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

hist = model.fit(x_train, y_train, epochs=500, batch_size=1, validation_split=0.2, callbacks=[es, mcp])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(xp_ohe)
print(y_predict)              # float 형
# print(y_predict.shape)      # (10000, 10)

y_predict = np.round(y_predict)
print("태운이 참 재미없다. : ", y_predict)

'''
loss :  0.3371948301792145
accuracy :  0.75
[[0.96953964]]
태운이 참 재미없다. :  [[1.]]

loss :  0.20599454641342163 / accuracy :  1.0 > k65_06_date_0819.2017_epo_0023_valloss_0.1251
[[0.93224716]]
태운이 참 재미없다. :  [[1.]]

'''
