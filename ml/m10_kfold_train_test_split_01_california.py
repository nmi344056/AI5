import numpy as np
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)
# print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123)

scaler = RobustScaler()
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
194/194 [==============================] - 0s 250us/step - loss: 0.5225
loss :  0.522547721862793
r2 score :  0.6048158187177399

++++++++++++++++++++
loss :  0.7035481333732605
r2 score :  0.4679315063315159

SVR
acc : [-0.01690779 -0.02598271 -0.0317809  -0.01445799 -0.02857462] avg acc : -0.0235
time :  39.51 초

train_test_split
acc : [0.65990359 0.67591959 0.68871773 0.66506714 0.68730051] avg acc : 0.6754
cross_val_predict : 0.6728408487474815
time :  22.63 초

'''
