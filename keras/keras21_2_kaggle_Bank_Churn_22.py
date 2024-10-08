# https://www.kaggle.com/competitions/playground-series-s4e1

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\playground-series-s4e1\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape)      # (165034, 13)
print(test_csv.shape)       # (110023, 12)
print(mission_csv.shape)    # (110023, 1)

print(train_csv.columns)
# Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited'],

print(train_csv.isnull().sum())     # 결측치가 없다
print(test_csv.isnull().sum())

encoder = LabelEncoder()
train_csv['Geography'] = encoder.fit_transform(train_csv['Geography'])
test_csv['Geography'] = encoder.fit_transform(test_csv['Geography'])
train_csv['Gender'] = encoder.fit_transform(train_csv['Gender'])
test_csv['Gender'] = encoder.fit_transform(test_csv['Gender'])

x = train_csv.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
print(x)                            # [165034 rows x 10 columns]
y = train_csv['Exited']
print(y.shape)                      # (165034,)

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

print(np.unique(y, return_counts=True))     
# (array([0, 1], dtype=int64), array([130113,  34921], dtype=int64))
print(pd.DataFrame(y).value_counts())
# 0         130113
# 1          34921

from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
x[:] = scalar.fit_transform(x[:])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1866)

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=10))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
start_time = time.time()
es = EarlyStopping(monitor='val_loss', mode = 'min', patience=10, restore_best_weights=True,)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[es])
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
print(y_predict[:20])               # y' 결과
y_predict = np.round(y_predict)
print(y_predict[:20])               # y' 반올림 결과

accuracy_score = accuracy_score(y_test, y_predict)
print("acc score : ", accuracy_score)
print("time : ", round(end_time - start_time, 2), "초")

y_submit = model.predict(test_csv)
print(y_submit.shape)               # (110023, 1)

y_submit = np.round(y_submit)
mission_csv['Exited'] = y_submit
mission_csv.to_csv(path + "sample_submission_0722_19.csv")

'''
32 64 128 128 128 64 32 1 / train_size=0.9, random_state=1866 / patience=10 / epochs=1000, batch_size=8
loss :  0.09958957880735397 / accuracy :  0.863 > 0.49996

32 16 16 16 16 1 / train_size=0.8, random_state=3434 / patience=10 / epochs=100, batch_size=16
loss :  0.09855161607265472 / accuracy :  0.867 > 0.41899

++++++++++++++++++++++++++++++
random_state=6666
random_state=1866
random_state=1186






'''
