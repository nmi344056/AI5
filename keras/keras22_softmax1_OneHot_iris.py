import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
datasets = load_iris()
# # print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (150, 4) (150,)

print(y)    
print(np.unique(y, return_counts=True))     # (array([0, 1, 2]), array([50, 50, 50], dtype=int64))
print(pd.value_counts(y))
# 0    50
# 1    50
# 2    50

# 케라스, 사이킷런, 판다스 세개 다 찾기

# 케라스
# from tensorflow.keras.utils import to_categorical
# y_ohe = to_categorical(y)
# print(y_ohe)
# print(y_ohe.shape)

# 사이킷런
# from sklearn.preprocessing import OneHotEncoder     # 전처리
# y_ohe3 = y.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)    # True가 디폴트 
# y_ohe3 = ohe.fit_transform(y_ohe3)   # -1 은 데이터 수치의 끝 
#                                 # sklearn의 문법 = 행렬로 주세요, reshape 할때 데이터의 값과 순서가 바뀌면 안된다.
# print(y_ohe3)

#  판다스
# y_ohe2 = pd.get_dummies(y) 
# print(y_ohe2)
# print(y_ohe2.shape)   # (150, 3)



x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8,
                                                      random_state=315, stratify=y)
print(x_train.shape, y_train.shape)     # (120, 4) (120,)
print(x_test.shape, y_test.shape)       # (30, 4) (30,)

print(pd.value_counts(y_train)) 

#2. 모델구성
model = Sequential()
model.add(Dense(8, activation='relu', input_dim=4))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))

model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 20,
    restore_best_weights=True
)
hist = model.fit(x_train, y_train, epochs=500, batch_size=24,
                 validation_split=0.2, callbacks=[es])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test,
                      verbose=1)
print('로스 :', loss)
print('acc :', round(loss[1],3))

y_pred = model.predict(x_test)
print(y_pred)
y_pred = np.round(y_pred)
print(y_pred)

accuracy_score = accuracy_score(y_test, y_pred)
print('acc_score :', accuracy_score)
print('걸린시간 :', round(end - start , 2), '초')

# 로스 : [0.06856824457645416, 1.0]
# acc : 1.0
