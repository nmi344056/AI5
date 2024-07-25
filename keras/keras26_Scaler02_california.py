# keras19_EarlyStopping2_california copy

import numpy as np
import sklearn as sk
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
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

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience=10,
    restore_best_weights=True,
)
hist = model.fit(x_train, y_train,
          validation_split=0.25, epochs=500, batch_size=100,
          callbacks=[es])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)
print("걸린시간 :", round(end-start, 2),"초")     # round 2 는 반올림자리

# 로스 :  0.5764865875244141
# r2스코어 :  0.572165479075531

# print("================== hist ==============")
# print(hist)
# print("================== hist.history ==============")
# print(hist.history)
# print("================== hist ==============")
# print(hist.history['loss'])
# print("================== hist ==============")
# print(hist.history['val_loss'])
# print("===================================")

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] ='Malgun Gothic'    # 한글깨짐 해결
# plt.figure(figsize=(9,6))       #그림판 사이즈
# plt.plot(hist.history['loss'], c='red', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# plt.legend(loc='upper right')
# plt.title('캘리포니아 loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()
# plt.show()


# MinMaxScaler 적용후
# 로스 :  0.5007096529006958
# r2스코어 :  0.6284026402135778
# 걸린시간 : 2.56 초

# StandardScaler
# 로스 :  0.49646610021591187
# r2스코어 :  0.6315520144115021
# 걸린시간 : 2.77 초

# MaxAbsScaler
# 로스 :  0.5135042071342468
# r2스코어 :  0.6189073210794624
# 걸린시간 : 10.02 초

# RobustScaler   로스의 폭이 크다 0.51까지 나옴
# 로스 :  0.48958998918533325
# r2스코어 :  0.6366550315381421
# 걸린시간 : 3.85 초