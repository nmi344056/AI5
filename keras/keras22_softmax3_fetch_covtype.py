from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#1. 데이터 
datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))     # (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],dtype=int64))

# one hot encoding
y_ohe = pd.get_dummies(y)
print(y_ohe.shape)  # (581012, 7)

print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5, random_state=5353,
                                                    stratify=y)    # stratify = train_size 에 맞게 정확하게 잘라준다. 분류에서는 넣어주면 성능이 좋다.
                                                     # stratify = train_size 에 맞게 정확하게 잘라준다. 분류에서는 넣어주면 성능이 좋다.

print(x_train.shape, y_train.shape)     # (522910, 54) (522910,)
print(x_test.shape, y_test.shape)       # (58102, 54) (58102,)

print(pd.value_counts(y_train))
# 2    254915
# 1    190784
# 3     32088
# 7     18439
# 6     15623
# 5      8575
# 4      2486

#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=54, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=60,
    restore_best_weights=True
)
model.fit(x_train, y_train, epochs=1000, batch_size=500,
          verbose=1,
          validation_split=0.2,
          callbacks=[es]
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :',loss[0])
print('acc :',round(loss[1],4))

y_pre = model.predict(x_test)
r2 = r2_score(y_test, y_pre)
print('r2 score :', r2)

accuracy_score = accuracy_score(y_test,np.round(y_pre))
print('acc_score :', accuracy_score)
print('걸린 시간 :', round(end-start, 2), '초')


