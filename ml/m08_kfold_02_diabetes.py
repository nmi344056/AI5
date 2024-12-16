import numpy as np
import time
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR

#1. 데이터
x, y = load_diabetes(return_X_y=True)
print(x)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델 구성
model = SVR()

start = time.time()
scores = cross_val_score(model, x, y, cv=kfold)
end = time.time()

print('acc :', scores, 'avg acc :', round(np.mean(scores), 4))
print("time : ", round(end - start, 2), "초")

'''
loss :  2197.28173828125
r2 sorce :  0.596755557322737

loss :  2156.3740234375
r2 sorce :  0.6042629302834428

SVR
acc : [0.18093916 0.15563965 0.15939131 0.15210134 0.14966275] avg acc : 0.1595
time :  0.03 초

'''
