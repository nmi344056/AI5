import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape, y.shape)     # (13, 3) (13,)

# [실습] x_predict = np.array([50,60,70])   -> 80 출력


# x = x.reshape(7, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)              # (7, 3, 1)
# 3-D tensor with shape (batch_size, timesteps, feature).

#2. 모델 구성
model = Sequential()
model.add(LSTM(16, input_shape=(3,1)))        # 3개 성능차이 있다. 골라서 사용
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True,)

# ########## mcp 세이브 파일명 만들기 시작 ##########
# import datetime
# date = datetime.datetime.now()
# print(date)
# print(type(date))

# date = date.strftime("%m%d.%H%M")
# print(date)
# print(type(date))

# path = './_save/keras52/'
# filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
# filepath = "".join([path, 'k52_02_date_', date, '_epo_', filename])

# ########## mcp 세이브 파일명 만들기 끝 ##########

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

# hist = model.fit(x, y, epochs=100, batch_size=10, validation_split=0.00001, callbacks=[es, mcp])

hist = model.fit(x, y, epochs=100, batch_size=1)

model.save("./_save/keras52/k52_02.h5")

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)

x_predict = np.array([50,60,70]).reshape(1,3,1)
print(x_predict.shape)      # (1, 3, 1)
    
y_predict = model.predict(x_predict)

print('[50,60,70]의 결과 : ', y_predict)

'''
[실습] 80.0 만들기
[8,9,10]의 결과 :  [[79.11273]]

74~75(?) 정도가 좋다, 요구사항에 맞추기위함일 뿐

'''
