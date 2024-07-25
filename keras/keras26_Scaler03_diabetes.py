# keras19_EarlyStopping3_diabetes copy

import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)     #(442, 10) (442,)

#[실습] 맹그러봐
# R2 0.62 이상

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state= 8000)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
model.add(Dense(16, input_dim = 10))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=50,
    restore_best_weights=True,

)
hist = model.fit(x_train, y_train, validation_split=0.2,
           epochs=1000, batch_size=32,
           callbacks=[es])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 스코어: ", r2)
print("걸린시간 :", round(end-start, 2), "초")

# print("============================= hist =========================")
# print(hist)
# print("============================= hist.history ==================")
# print(hist.history)
# print("============================= loss ==================")
# print(hist.history['loss'])
# print("============================= val_loss ==================")
# print(hist.history['val_loss'])
# print("==========================================================")

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] ='Malgun Gothic'
# plt.figure(figsize=(9,6))       # 그림판 사이즈
# plt.plot(hist.history['loss'], c='red', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# plt.legend(loc='upper right')
# plt.title('당뇨 loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

# MinMaxScaler
# 로스 :  2291.559326171875
# r2 스코어:  0.6161708204975695
# 로스 :  2285.40478515625
# r2 스코어:  0.6172017509215408

# StandardScaler
# 로스 :  2548.71142578125
# r2 스코어:  0.6113628098257959
# 걸린시간 : 2.7 초

# MaxAbsScaler
# 로스 :  2518.020263671875
# r2 스코어:  0.6160426723925053
# 걸린시간 : 1.58 초

# RobustScaler
# 로스 :  2530.133056640625
# r2 스코어:  0.6141957245895928
# 걸린시간 : 3.73 초