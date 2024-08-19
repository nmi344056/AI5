import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=1000,
    # maxlen=50,
    # test_split=0.2,
)

# print(x_train, x_train.shape)     # (25000,)
# print(x_test, x_test.shape)       # (25000,)
# print(y_train, y_train.shape)     # [1 0 0 ... 0 1 0] (25000,)
# print(y_test, y_test.shape)       # [0 1 1 ... 0 0 0] (25000,)

print(np.unique(y_train))           # [0 1]
print(len(np.unique(y_train)))      # 2

# 넘파이 안에 리스트가 들어가있다. -> 데이터의 길이가 일정하지 않을 수 있다. -> 길이 조절 필요
print(type(x_train))                # <class 'numpy.ndarray'>
print(type(x_train[0]))             # <class 'list'>, 넘파이로 변환 필요
print(len(x_train[0]), len(x_train[1])) # 218 189

# 이 데이터의 가장 긴 문장, 짧은 문장, 평균
print('imdb의 최대 길이 : ', max(len(i) for i in x_train))           # 2494
print('imdb의 최소 길이 : ', min(len(i) for i in x_train))           # 11
print('imdb의 평균 길이 : ', sum(map(len, x_train)) / len(x_train))  # 238.71364

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')

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
model.add(Dense(1, activation='sigmoid'))

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

path = './_save/keras66/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k66_02_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es, mcp])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

'''
loss :  0.3816567063331604 / accuracy :  0.829 > k66_02_date_0820.1447_epo_0003_valloss_0.3885

relu 추가 (오래걸린다)
loss :  0.5753527283668518 / accuracy :  0.691

'''
