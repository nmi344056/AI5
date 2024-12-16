import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, shuffle=True, random_state=123
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. 모델 구성
model = SVC()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('acc :', scores, '\navg acc :', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test)          # cv=kfold, Default : 5
# print(y_predict)
# print(y_test)
'''
[1 0 2 2 0 0 2 1 2 0 0 1 2 1 2 1 0 0 0 0 0 1 1 2 2 1 1 1 1 1]
[1 0 2 2 0 0 2 1 2 0 0 1 2 1 2 1 0 0 0 0 0 2 2 1 2 2 1 1 1 1]
'''

acc = accuracy_score(y_test, y_predict)
print('cross_val_predict :', acc)

'''
KFold
acc : [1.         0.86666667 1.         0.96666667 0.96666667] 
avg acc : 0.96

StratifiedKFold
acc : [0.93333333 0.96666667 0.93333333 1.         1.        ] 
avg acc : 0.9667

train_test_split / StratifiedKFold
acc : [0.95833333 0.95833333 0.95833333 1.         1.        ] 
avg acc : 0.975
cross_val_predict : 0.8666666666666667

'''
