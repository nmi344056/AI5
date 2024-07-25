# https://dacon.io/competitions/open/235576/data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

#1. 데이터
path = "./_data/dacon/따릉이/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)            # [1459 rows x 11 columns] / [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)             # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
print(submission_csv)       # [715 rows x 1 columns]

print(train_csv.shape)      # (1459, 10)
print(test_csv.shape)       # (715, 9)
print(submission_csv.shape) # (715, 1)

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

print(train_csv.info())

########## 결측치 처리 1. 삭제 ##########
# print(train_csv.isnull().sum())
print(train_csv.isna().sum())

train_csv = train_csv.dropna()
print(train_csv.isna().sum())
print(train_csv)            # [1328 rows x 10 columns]
print(train_csv.info())

print(test_csv.info())

test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())

x = train_csv.drop(['count'], axis=1)
print(x)                    # [1328 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape)              # (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=4343)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)       # 추가

print(x_train)
print(np.min(x_train), np.max(x_train))     # 0.0 1.0
print(np.min(x_test), np.max(x_test))       # 0.0 1.0

#2. 모델구성
model = Sequential()
model.add(Dense(126, input_shape=(9,)))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=500, batch_size=24, validation_split=0.2, callbacks=[es])
end = time.time()

#4. 평가, 예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)
print("time : ", round(end - start, 2), "초")

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)           # (715, 1)

########## submission.csv 만들기 (count컬럼에 값만 넣으면 된다) ##########
submission_csv['count'] = y_submit
print(submission_csv)           # [715 rows x 1 columns]
print(submission_csv.shape)     # (715, 1)

submission_csv.to_csv(path + "submission_0725_18.csv")

print("loss : ", loss)
print("r2 score : ", r2)

'''
126 64 32 32 32 1 / train_size=0.9, random_state=4343 / epochs=500, batch_size=24
                loss :  1753.6702880859375 / r2 score :  0.6841434676973406
MinMaxScaler > loss :  1637.28759765625 / r2 score :  0.7051053444572746 > best
StandardScaler > loss :  1675.6771240234375 / r2 score :  0.6981909408225387
MaxAbsScaler > loss :  1641.0303955078125 / r2 score :  0.7044312138913198 > 2
RobustScaler > loss :  1651.0953369140625 / r2 score :  0.7026184262609562
'''
