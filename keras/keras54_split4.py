# 54_3 copy
# [실습] (N, 10, 1) -> (N, 5, 2)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))    # y=106

# [실습] 101부터 107을 찾아라

size = 11

print(a.shape)  # (100,)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
# print(bbb)
print(bbb.shape)        # (90, 11)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)

print(x.shape, y.shape) # (90, 10) (90,)

x = x.reshape(1,5,2)

#2. 모델 구성
model = Sequential()
model.add(LSTM(16, input_shape=(10,1)))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x, y, epochs=100, batch_size=1)

model.save("./_save/keras54/k54_03.h5")

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)

x_predict = np.array(range(96, 106)).reshape(1,10,1)

y_predict = model.predict(x_predict)
print(y_predict.shape)      # (1, 3, 1)
    
print('예상 결과 106 : ', y_predict)

'''
loss :  0.10763903707265854
예상 결과 106 :  [[103.30599]]

'''
