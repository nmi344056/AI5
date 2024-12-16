import numpy as np
import time
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
x, y = load_diabetes(return_X_y=True)
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

start = time.time()
scores = cross_val_score(model, x_train, y_train, cv=kfold)
end = time.time()

print('acc :', scores, 'avg acc :', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test)
r2 = r2_score(y_test, y_predict)
print('cross_val_predict :', r2)

print("time : ", round(end - start, 2), "초")

'''
loss :  2197.28173828125
r2 sorce :  0.596755557322737

loss :  2156.3740234375
r2 sorce :  0.6042629302834428

SVR
acc : [0.18093916 0.15563965 0.15939131 0.15210134 0.14966275] avg acc : 0.1595
time :  0.03 초

train_test_split
acc : [0.04610951 0.04296415 0.11195374 0.08379547 0.08982703] avg acc : 0.0749
cross_val_predict : -0.03948304536503011
time :  0.02 초

'''
