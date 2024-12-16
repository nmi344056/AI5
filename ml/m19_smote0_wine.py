import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as XGBClassifier

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']

print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
print(y)
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
 '''

x = x[:-40]
y = y[:-40]

print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71,  8], dtype=int64))
print(y)
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2]
 불균형 데이터가 됐다.
 '''

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, stratify=y, shuffle=True, random_state=333)

#2. 모델 구성
model = XGBClassifier()

#3. 컴파일, 훈련
model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose=True)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('model.score :', results)

# 지표 : f1_score
y_predict = model.predict(x_test)
print('f1_score :', f1_score(y_test, y_predict, average='macro'))


'''
에러 해결하기
  File "c:\ai5\study\ml\m19_smote1_wine.py", line 49, in <module>
    model = XGBClassfier()
TypeError: 'module' object is not callable
'''
