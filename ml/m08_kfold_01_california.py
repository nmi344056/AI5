import numpy as np
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)
print(x)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델 구성
model = SVR()

#3. 훈련
start = time.time()
scores = cross_val_score(model, x, y, cv=kfold)
end = time.time()

print('acc :', scores, 'avg acc :', round(np.mean(scores), 4))
print("time : ", round(end - start, 2), "초")

'''
194/194 [==============================] - 0s 250us/step - loss: 0.5225
loss :  0.522547721862793
r2 score :  0.6048158187177399

++++++++++++++++++++
loss :  0.7035481333732605
r2 score :  0.4679315063315159

SVR
acc : [-0.01690779 -0.02598271 -0.0317809  -0.01445799 -0.02857462] avg acc : -0.0235
time :  39.51 초

'''
