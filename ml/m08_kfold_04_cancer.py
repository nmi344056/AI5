import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)
# print(x)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델 구성
model = SVR()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)
print('acc :', scores, '\navg acc :', round(np.mean(scores), 4))

'''
63 32 32 32 32 1 / train_size=0.8, random_state=555 / epochs=100, batch_size=8 / verbose=1
mse / loss : 

loss :  0.10229721665382385
accuracy :  0.974

SVR
acc : [0.76597921 0.73488435 0.76122965 0.74312846 0.64602358] 
avg acc : 0.7302

'''
