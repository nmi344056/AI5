# 23_1 copy

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# 1. 데이터
datasets = load_breast_cancer()      # feature_name 때문에
x = datasets.data
y = datasets.target

x = pd.DataFrame(x, columns=[datasets.feature_names])
print(x.shape)  # (569, 30)

random_state1=1223
random_state2=1223

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, random_state=random_state1)

#2. 모델 구성
model = DecisionTreeClassifier(random_state=random_state2)

print('random_state :', random_state1, random_state2)

model.fit(x_train, y_train)
print('==========', model.__class__.__name__, '==========')
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)

'''
random_state : 1223 1223
========== DecisionTreeClassifier ==========
acc : 0.9473684210526315
[0.         0.05030732 0.         0.         0.         0.
 0.         0.         0.         0.0125215  0.         0.03023319
 0.         0.         0.         0.         0.00785663 0.
 0.         0.         0.72931244 0.         0.0222546  0.01862569
 0.01611893 0.         0.         0.0955152  0.01725451 0.        ]
'''

########## 하위 20~25% 컬럼 제거 ##########

x2 = x
percentiles = np.percentile(model.feature_importances_, 20)

for i, fi in enumerate(model.feature_importances_):
    if fi <= percentiles:
       x2 = x2.drop(datasets.feature_names[i], axis=1)

print(x2.shape)  # (569, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x2, y, train_size=0.8, stratify=y, random_state=random_state1)

# model = DecisionTreeClassifier(random_state=random_state2)

# print('random_state :', random_state1, random_state2)

model.fit(x_train, y_train)
print('==========', model.__class__.__name__, 'DROP ==========')
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)

'''
========== DecisionTreeClassifier DROP ==========
acc : 0.9473684210526315
[0.05030732 0.         0.03023319 0.00883871 0.74977167 0.
 0.02648232 0.00907558 0.12529121 0.        ]

결과 : 성능 동일

'''
