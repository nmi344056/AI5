import numpy as np
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
x, y = load_digits(return_X_y=True)
print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델 구성
model = SVC()

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
[실습] accuracy :  1.0 이상
128 256 256 256 128 10 / train_size=0.9, random_state=6666 / epochs=100, batch_size=100

loss :  0.11226184666156769
accuracy :  0.978

SVC
acc : [0.98888889 0.98055556 0.97771588 0.99164345 0.99442897] avg acc : 0.9866
time :  0.16 초

StratifiedKFold
acc : [0.98333333 0.98611111 0.99164345 0.99164345 0.98328691] avg acc : 0.9872
time :  0.16 초

train_test_split
acc : [0.95833333 0.94791667 0.95818815 0.96864111 0.96515679] avg acc : 0.9596
cross_val_predict : 0.6638150983040673
time :  0.17 초

'''
