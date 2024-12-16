import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import reuters
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000,
    # maxlen=50,
    test_split=0.2,
)

# print(x_train, x_train.shape)     # (8982,)
# print(x_test, x_test.shape)       # (2246,)
# print(y_train, y_train.shape)     # [ 3  4  3 ... 25  3 25] (8982,)
# print(y_test, y_test.shape)       # [ 3 10  1 ...  3  3 24] (2246,)

print(np.unique(y_train))           # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
print(len(np.unique(y_train)))      # 46

# 넘파이 안에 리스트가 들어가있다. -> 데이터의 길이가 일정하지 않을 수 있다. -> 길이 조절 필요
print(type(x_train))                # <class 'numpy.ndarray'>
print(type(x_train[0]))             # <class 'list'>, 넘파이로 변환 필요
print(len(x_train[0]), len(x_train[1])) # 87 56

# 이 데이터의 가장 긴 문장, 짧은 문장, 평균
print('뉴스기사의 최대 길이 : ', max(len(i) for i in x_train))           # 2376
print('뉴스기사의 최소 길이 : ', min(len(i) for i in x_train))           # 13
print('뉴스기사의 평균 길이 : ', sum(map(len, x_train)) / len(x_train))  # 145.5398574927633

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')

# [실습] 만들기, y 원핫
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train, y_train.shape)       # (8982, 46)
print(y_test, y_test.shape)         # (2246, 46)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# print(x_train)
# print(np.min(x_train), np.max(x_train))     # 0.0 1.0
# print(np.min(x_test), np.max(x_test))       # -0.0020060180541624875 1.0010020040080159

#2. 모델 구성
model = Sequential()
model.add(Embedding(1000, 64))
model.add(LSTM(128))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(46, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))

date = date.strftime("%m%d.%H%M")
print(date)
print(type(date))

path = './_save/keras66/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k66_01_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es, mcp])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

'''
loss :  1.4338968992233276
accuracy :  0.656

'''
