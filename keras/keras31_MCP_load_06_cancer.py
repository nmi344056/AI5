from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import time

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(np.unique(y))                         # [0 1] 꼭 확인
print(np.unique(y, return_counts=True))     # (array([0, 1]), array([212, 357], dtype=int64))

print(type(x))                              # <class 'numpy.ndarray'>
print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212

# print(y.value_counts())                   # AttributeError: 'numpy.ndarray' object has no attribute 'value_counts'

print(pd.Series(y))
print("++++++++++++++++++++")
print(pd.Series(y).value_counts())
print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=555)

scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))     # 0.0 1.0
print(np.min(x_test), np.max(x_test))       # -0.0008218036468019101 1.5452539790933444

print(x_train.shape, y_train.shape)         # (455, 30) (455,)
print(x_test.shape, y_test.shape)           # (114, 30) (114,)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(63, activation='relu', input_shape=(30,)))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# #3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

# start = time.time()
# hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es, mcp])
# end = time.time()

#4. 평가, 예측
print('========== 2. mcp 출력 ==========')
model = load_model('./_save/keras30_mcp/k30_06_cancer_date_0726.2040_epo_0032_valloss_17.7942.hdf5')
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
print(y_predict[:20])               # y' 결과
y_predict = np.round(y_predict)
print(y_predict[:20])               # y' 반올림 결과

accuracy_score = accuracy_score(y_test, y_predict)
print("acc score : ", accuracy_score)

print("===============")
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

'''
63 32 32 32 32 1 / train_size=0.8, random_state=555 / epochs=100, batch_size=8 / verbose=1
mse / loss : 


'''
