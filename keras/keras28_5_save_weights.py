# [복사] keras26_Scaler01_boston.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston
import numpy as np
import time

#1. 데이터
dataset = load_boston()
# print(dataset)
x = dataset.data
y = dataset.target

print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=555)
print(x)
print(x.shape)             # (506, 13)
print(y)
print(y.shape)             # (506,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))     # 0.0 1.0
print(np.min(x_test), np.max(x_test))       # -0.06211435623200334 1.0280851063829786 (StandardScaler)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(5))
model.add(Dense(1))

# model.save("./_save/keras28/keras28_1_save_model.h5")
model.save_weights("./_save/keras28/keras28_5_save_weight1.h5")

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.2)
end = time.time()

# model.save("./_save/keras28/keras28_3_save_model.h5")
model.save_weights("./_save/keras28/keras28_5_save_weight2.h5")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)
# print("time : ", round(end - start, 2), "초")
