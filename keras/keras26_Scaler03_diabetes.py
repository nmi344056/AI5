from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import numpy as np
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)          # (442, 10) (442,)

#[실습]만들기 R2 성능 0.62 이상

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=999)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))     # 0.0 1.0
print(np.min(x_test), np.max(x_test))       # -0.06806282722513224 1.0

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_shape=(10,)))
model.add(Dense(75))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2, callbacks=[es])
end = time.time()

#4.평가, 예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 sorce : ", r2)
print("time : ", round(end - start, 2), "초")

'''
100 75 50 30 1 / train_size=0.8, random_state=999 / epochs=100, batch_size=10
                loss :  2159.803955078125 / r2 sorce :  0.6036334555474677
MinMaxScaler > loss :  2125.1162109375 / r2 sorce :  0.6099993399059593 > 2
StandardScaler > loss :  2133.20263671875 / r2 sorce :  0.60851531739284
MaxAbsScaler > loss :  2088.937744140625 / r2 sorce :  0.6166388284587843 > best
RobustScaler > loss :  2136.62158203125 / r2 sorce :  0.6078878932536853
'''
