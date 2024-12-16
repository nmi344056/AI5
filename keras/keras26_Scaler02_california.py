from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
import numpy as np
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)         # (20640, 8) (20640,)

#[실습] 만들기 R2 성능 0.59 이상
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))     # 0.0 1.0000000000000002
print(np.min(x_test), np.max(x_test))       # -0.005005005005005003 1.0

#2. 모델구성
model = Sequential()
model.add(Dense(128, input_shape=(8,)))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, callbacks=[es])
end = time.time()

#4. 평가,예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)
print("time : ", round(end - start, 2), "초")

'''
128 64 32 32 32 1 / train_size=0.7, random_state=123 / epochs=100, batch_size=100
                loss :  0.702932596206665 / r2 score :  0.4683970720451244
MinMaxScaler > loss :  0.5462954044342041 / r2 score :  0.586856315595612
StandardScaler > loss :  0.5269249081611633 / r2 score :  0.6015055763433855 > 2
MaxAbsScaler > loss :  0.6288350820541382 / r2 score :  0.5244345163252213
RobustScaler > loss :  0.5263957977294922 / r2 score :  0.6019057841014295 > best
'''
