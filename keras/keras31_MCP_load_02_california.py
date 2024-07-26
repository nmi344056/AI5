# keras19_EarlyStopping2_california copy

import numpy as np
import sklearn as sk
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9,
                                                    random_state=3)
print('x_train :', x_train)
print('x_test :', x_test)
print('y_train :', y_train)
print('y_test :', y_test)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()



scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print('x_train :', x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

# #2. 모델구성
# model = Sequential()
# model.add(Dense(10, input_dim=8))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(1))

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# start = time.time()

# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(
#     monitor = 'val_loss',
#     mode = 'min',
#     patience=10,
#     restore_best_weights=True,
# )
# hist = model.fit(x_train, y_train,
#           validation_split=0.25, epochs=500, batch_size=100,
#           callbacks=[es])
# end = time.time()

print("======================= 2. MCP 출력 ========================")
model = load_model('./_save/keras30_mcp/02_california/k30_0726_1914_0030-0.5338.hdf5')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)
# print("걸린시간 :", round(end-start, 2),"초")     # round 2 는 반올림자리
