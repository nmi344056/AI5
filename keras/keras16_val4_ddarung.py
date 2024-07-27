# https://dacon.io/competitions/open/235576/data

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
path = "./_data/따릉이/"

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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.87, random_state=4343)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=9))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=23, validation_split=0.2)

#4. 평가, 예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)

'''
100 50 30 10 1 / train_size=0.82, random_state=4343 / epochs=1000, batch_size=30 / loss :  2054.7568359375
100 50 30 10 1 / train_size=0.9, random_state=4343 / epochs=1000, batch_size=30 / loss :  1697.7445068359375 (제출1)
81 27 45 18 1 / train_size=0.9, random_state=4343 / epochs=1000, batch_size=15 / loss :  1691.560791015625 (제출2)
100 500 300 100 1 / train_size=0.9, random_state=4343 / epochs=700, batch_size=3 / loss :  1688.20947265625 (제출3)
100 50 30 10 1 / train_size=0.982, random_state=4343 / epochs=1000, batch_size=30 / loss :  1356.6510009765625 (제출4)
200 100 50 30 10 1 / train_size=0.982, random_state=4343 / epochs=1000, batch_size=30 / loss :  1336.6988525390625
200 100 50 30 10 30 1 / train_size=0.98, random_state=4343 / epochs=1000, batch_size=30 / loss :  1601.2872314453125 (제출5)
200 100 50 30 10 1 / train_size=0.9, random_state=5757 / epochs=1000, batch_size=30 / loss :  2201.14892578125
200 100 50 30 10 1 / train_size=0.9, random_state=512 / epochs=1000, batch_size=30 / loss :  2556.475830078125 (제출6)
30 20 10 1 / train_size=0.9, random_state=123 / epochs=700, batch_size=30 / loss :  2347.8154296875 (제출7)
3 3 3 3 1 / train_size=0.8, random_state=4343 / epochs=700, batch_size=50 / loss :  2222.330078125 (제출8)
100 50 30 10 1 / train_size=0.82, random_state=4343 / epochs=1000, batch_size=30 / loss :  2055.451904296875 (제출9)
100 50 30 1 / train_size=0.87, random_state=4343 / epochs=1000, batch_size=23 / loss :  1921.1092529296875 (제출10)
'''

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)           # (715, 1)

########## submission.csv 만들기 (count컬럼에 값만 넣으면 된다) ##########
submission_csv['count'] = y_submit
print(submission_csv)           # [715 rows x 1 columns]
print(submission_csv.shape)     # (715, 1)

submission_csv.to_csv(path + "submission_0716_1930.csv")

print("loss : ", loss)
