from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
import time

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (581012, 54) (581012,)

print(pd.value_counts(y))    
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747
# 문제 : 0이 없다, onehot을 0이 아닌 1부터 시작한다.

print(y)
print(np.unique(y, return_counts=True))

# from tensorflow.keras.utils import to_categorical
# y_ohe = to_categorical(y)
# print(y_ohe)
# print(y_ohe.shape)              # (581012, 8)

y_ohe = pd.get_dummies(y)          # pandas
print(y_ohe)                       # 1  2  3  4  5  6  7
print(y_ohe.shape)                 # (581012, 7)

# print("==============================")
# from sklearn.preprocessing import OneHotEncoder
# y_ohe = y.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)        # True가 default
# ohe.fit(y_ohe)
# y_ohe = ohe.transform(y_ohe)
# print(y_ohe)
# print(y_ohe.shape)                 # (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, train_size=0.9, random_state=6666,
                                                    stratify=y)

# print(pd.value_counts(y_train))
# 2    141429       141651
# 1    106155       105920
# 3     17958       17877
# 7     10262       10255
# 6      8613       8683
# 5      4711       4747
# 4      1378       1373

print(x_train.shape, x_test.shape)      # (522910, 54) (58102, 54)
print(y_train.shape, y_test.shape)      # (522910, 8) (58102, 8)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# print(x_train)
# print(np.min(x_train), np.max(x_train))     # 0.0 1.0
# print(np.min(x_test), np.max(x_test))       # -0.009150326797385616 1.0026143790849673

#2. 모델구성
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(54,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(126, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=300, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
print(y_predict[:20])
y_predict = np.round(y_predict)
print(y_predict[:20])

accuracy_score = accuracy_score(y_test, y_predict)
print("acc score : ", accuracy_score)

print(x)
'''
128 256 256 256 126 8 / train_size=0.9, random_state=6666 / epochs=100, batch_size=300, validation_split=0.2

MinMaxScaler > sssss

StandardScaler > 

MaxAbsScaler > 

RobustScaler > 





acc score :  0.9335478985232867
++++++++++++++++++++++++++++++
MinMaxScaler / acc score :  0.9406655972682147
StandardScaler / acc score :  0.9504313163927768
'''
