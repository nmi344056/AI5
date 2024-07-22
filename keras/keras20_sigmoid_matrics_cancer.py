import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)   # (569, 30) (569,)

print(np.unique(y, return_counts=True))     
# (array([0, 1]), array([212, 357], dtype=int64))  불균형 데이터인지 확인
print(type(x))  # <class 'numpy.ndarray'>  넘파이 파일

# print(y.value_counts())  # 에러
print(pd.DataFrame(y).value_counts())   # 넘파이를 판다스 데이터프레임으로 바꿔줘
# 1    357
# 0    212
print(pd.Series(y).value_counts())
pd.value_counts(y)          # 셋 다 똑같다.



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=315)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)




#2. 모델구성
model = Sequential()
model.add(Dense(40, activation='relu', input_dim=30))
model.add(Dense(50, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu')) #중간에 sigmoid 넣어줄수있다.
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'acc', 'mse'])  # 매트릭스에 애큐러시를 넣으면 반올림해준다.
start = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 20,
    restore_best_weights=True
)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=20,
                 validation_split=0.2,
                 callbacks=[es]
                 )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test,
                      verbose=1)
print('로스 : ', loss[0])
print("acc : ", round(loss[1],3))  # 애큐러시, 3자리 반올림 

y_pred = model.predict(x_test)
print(y_pred)
y_pred = np.round(y_pred)
print(y_pred)
from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_pred)  
# r2 = r2_score(y_test, y_predict)
print("acc_score : ", accuracy_score)
print("걸린시간 : ", round(end - start , 2),"초")

# 로스 :  [0.04040343314409256, 0.9473684430122375, 0.9473684430122375, 0.04040343314409256]
# 로스 :  [0.02593756467103958, 0.9649122953414917, 0.9649122953414917, 0.02593756467103958]
# acc :  0.965
# acc_score :  0.9824561403508771