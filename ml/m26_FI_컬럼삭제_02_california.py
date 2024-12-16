# 23_1 copy

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# 1. 데이터
datasets = fetch_california_housing()      # feature_name 때문에
x = datasets.data
y = datasets.target

x = pd.DataFrame(x, columns=[datasets.feature_names])
# print(x)
#        MedInc HouseAge  AveRooms AveBedrms Population  AveOccup Latitude Longitude
# 0      8.3252     41.0  6.984127  1.023810      322.0  2.555556    37.88   -122.23

random_state1=1223
random_state2=1223

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_state1)

#2. 모델 구성
model = XGBRegressor(random_state=random_state2)

print('random_state :', random_state1, random_state2)

model.fit(x_train, y_train)
print('==========', model.__class__.__name__, '==========')
print('r2 :', model.score(x_test, y_test))
print(model.feature_importances_)

'''
random_state : 1223 1223
========== XGBRegressor ==========
r2 : 0.8384930657222394
[0.49375907 0.06520814 0.04559402 0.02538511 0.02146595 0.14413244 0.0975963  0.10685894]
'''

########## 하위 20~25% 컬럼 제거 ##########

x2 = x
percentiles = np.percentile(model.feature_importances_, 25)

for i, fi in enumerate(model.feature_importances_):
    if fi <= percentiles:
       x2 = x2.drop(datasets.feature_names[i], axis=1)

# print(x2)
#        MedInc HouseAge  AveRooms  AveOccup Latitude Longitude
# 0      8.3252     41.0  6.984127  2.555556    37.88   -122.23

x_train, x_test, y_train, y_test = train_test_split(
    x2, y, train_size=0.8, random_state=random_state1)

#2. 모델 구성
model = XGBRegressor(random_state=random_state2)

print('random_state :', random_state1, random_state2)

model.fit(x_train, y_train)
print('==========', model.__class__.__name__, 'DROP ==========')
print('r2 :', model.score(x_test, y_test))
print(model.feature_importances_)

'''
random_state : 1223 1223
========== XGBRegressor DROP ==========
r2 : 0.8424404298528521
[0.51401776 0.06747776 0.05457867 0.15093587 0.1003058  0.1126841 ]

결과 : 성능 향상

'''
