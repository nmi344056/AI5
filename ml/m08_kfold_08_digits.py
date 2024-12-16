import numpy as np
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR

#1. 데이터
x, y = load_digits(return_X_y=True)
print(x)

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델 구성
model = SVC()

#3. 훈련
start = time.time()
scores = cross_val_score(model, x, y, cv=kfold)
end = time.time()

print('acc :', scores, 'avg acc :', round(np.mean(scores), 4))
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

'''
