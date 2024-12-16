# 23_1 copy

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# 1. 데이터
datasets = load_wine()      # feature_name 때문에
x = datasets.data
y = datasets.target

x = pd.DataFrame(x, columns=[datasets.feature_names])
print(x.shape)  # (178, 13)

random_state1=1223
random_state2=1223

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, random_state=random_state1)

#2. 모델 구성
model = RandomForestClassifier(random_state=random_state2)
# model4 = XGBClassifier(random_state=random_state2)

print('random_state :', random_state1, random_state2)

model.fit(x_train, y_train)
print('==========', model.__class__.__name__, '==========')
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)

'''
random_state : 1223 1223
========== RandomForestClassifier ==========
acc : 0.9444444444444444
[0.13789135 0.02251876 0.01336314 0.03826336 0.02830375 0.05255915
 0.14261827 0.00916645 0.03234439 0.13563367 0.07199803 0.13963923
 0.17570046]
'''

########## 하위 20~25% 컬럼 제거 ##########

x2 = x
percentiles = np.percentile(model.feature_importances_, 35)

for i, fi in enumerate(model.feature_importances_):
    if fi <= percentiles:
       x2 = x2.drop(datasets.feature_names[i], axis=1)

print(x2.shape)  # (178, 8)

x_train, x_test, y_train, y_test = train_test_split(
    x2, y, train_size=0.8, stratify=y, random_state=random_state1)

# model = RandomForestClassifier(random_state=random_state2)
# # model4 = XGBClassifier(random_state=random_state2)

# print('random_state :', random_state1, random_state2)

model.fit(x_train, y_train)
print('==========', model.__class__.__name__, 'DROP ==========')
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)

'''
========== RandomForestClassifier DROP ==========
acc : 0.9444444444444444
[0.19183985 0.02593047 0.02625342 0.04858361 0.14309247 0.01308727
 0.12161066 0.06577769 0.18572366 0.1781009 ]

결과 : 성능 향상

'''
