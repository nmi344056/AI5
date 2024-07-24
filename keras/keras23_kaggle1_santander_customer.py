# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

# [실습] 이진 분류 (다중분류로도 풀어보기)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import pandas as pd
import time

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\santander-customer-transaction-prediction\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)            # [200000 rows x 201 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)             # [200000 rows x 200 columns]

mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
print(mission_csv)          # [200000 rows x 1 columns]

print(train_csv.columns)    # Index(['target', 'var_0', 'var_1', ...

x = train_csv.drop(['target'], axis=1)
print(x)                    # [200000 rows x 200 columns]
y = train_csv['target']
print(y.shape)              # (200000,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=3,
                                                    stratify=y)

#2. 모델구성
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=200))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=300, batch_size=1000, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_pre = model.predict(x_test)
r2 = r2_score(y_test,y_pre)
print('r2 score :', r2)

accuracy_score = accuracy_score(y_test,np.round(y_pre))
print('acc_score :', accuracy_score)

### csv 파일 만들기 ###
y_submit = model.predict(test_csv)
print(y_submit)

y_submit = np.round(y_submit)
print(y_submit)

mission_csv['target'] = y_submit
mission_csv.to_csv(path + "sampleSubmission_0724_1605.csv")

print(mission_csv['target'].value_counts())