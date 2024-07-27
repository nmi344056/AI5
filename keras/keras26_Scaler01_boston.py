# [복사] keras25_input_shape.py

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
# model.add(Dense(100, input_dim=13))
model.add(Dense(32, input_dim=13,))    # 이미지는 input_shape=(8,8,1)
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=500, batch_size=10, validation_split=0.2, callbacks=[es])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)
print("time : ", round(end - start, 2), "초")

'''
32 16 16 16 1 / train_size=0.7, random_state=555 / epochs=500, batch_size=10
                loss :  23.239059448242188 / r2 score :  0.7301424718220688
MinMaxScaler > loss :  20.770709991455078 / r2 score :  0.7588055105996
StandardScaler > loss :  20.400049209594727 / r2 score :  0.7631097044958196 > best
MaxAbsScaler > loss :  22.515735626220703 / r2 score :  0.7385418668215343
RobustScaler > loss :  20.481996536254883 / r2 score :  0.7621581380586453 >2
'''
