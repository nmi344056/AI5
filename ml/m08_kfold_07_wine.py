import numpy as np
import time
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR

#1. 데이터
x, y = load_wine(return_X_y=True)
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
16 32 16 16 16 3 / train_size=0.9, random_state=666 / epochs=100, batch_size=1

loss :  0.0543439 / accuracy :  0.944

SVC
acc : [0.5        0.69444444 0.72222222 0.68571429 0.71428571] avg acc : 0.6633
time :  0.01 초

StratifiedKFold
acc : [0.72222222 0.72222222 0.61111111 0.62857143 0.74285714] avg acc : 0.6854
time :  0.01 초

'''
