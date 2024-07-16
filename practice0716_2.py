# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
path = "./_data/따릉이/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)

submission_csv = pd.read_csv(path + "submission.csv", index_col=0)
print(submission_csv)

print(train_csv.shape)
print(test_csv.shape)
print(submission_csv.shape)

print(train_csv.columns)

train_csv.info()
################### 결측치 처리 1. 삭제 ###############
print(train_csv.isna().sum())

train_csv = train_csv.dropna()
print(train_csv.isna().sum())
print(train_csv)
print(train_csv.isna().sum())
print(train_csv.info())

print(test_csv.info())

# test_csv 는 결측치 삭제 불가, test_csv 715와 submission 715 가 같아야한다
# 그래서 결측치 삭제하지 않고, 데이터의 평균값mean() 을 넣어준다.

test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())         # 다 715가 되었다.

x = train_csv.drop(['count'], axis=1)
print(x)
y = train_csv['count']
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.95,
                                                    random_state = 12015)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=500, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)

submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape)

submission_csv.to_csv(path + "submission_0716_2134.csv")

print('로스 :', loss)
print("r2 스코어 :", r2)



















