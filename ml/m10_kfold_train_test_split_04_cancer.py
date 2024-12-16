import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)
# print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123)

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델 구성
model = SVR()

#3. 훈련
start = time.time()
scores = cross_val_score(model, x_train, y_train, cv=kfold)
end = time.time()

print('acc :', scores, 'avg acc :', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test)
r2 = r2_score(y_test, y_predict)
print('cross_val_predict :', r2)

print("time : ", round(end - start, 2), "초")

'''
63 32 32 32 32 1 / train_size=0.8, random_state=555 / epochs=100, batch_size=8 / verbose=1
mse / loss : 

loss :  0.10229721665382385
accuracy :  0.974

SVR
acc : [0.76597921 0.73488435 0.76122965 0.74312846 0.64602358] avg acc : 0.7302

train_test_split
acc : [0.86906017 0.84449208 0.77634913 0.85780233 0.82269714] avg acc : 0.8341
cross_val_predict : 0.8119449331959998
time :  0.02 초

'''
