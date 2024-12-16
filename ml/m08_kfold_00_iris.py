import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

#1. 데이터
x, y = load_iris(return_X_y=True)
# print(x)

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

#2. 모델 구성
model = SVC()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)
print('acc :', scores, '\navg acc :', round(np.mean(scores), 4))

'''
KFold
acc : [1.         0.86666667 1.         0.96666667 0.96666667] 
avg acc : 0.96

StratifiedKFold
acc : [0.93333333 0.96666667 0.93333333 1.         1.        ] 
avg acc : 0.9667

'''
