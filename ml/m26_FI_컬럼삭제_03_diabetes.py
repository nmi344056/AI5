# 23_1 copy

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# 1. 데이터
datasets = load_diabetes()      # feature_name 때문에
x = datasets.data
y = datasets.target

x = pd.DataFrame(x, columns=[datasets.feature_names])
# print(x)
#           age       sex       bmi        bp        s1        s2        s3        s4        s5        s6
# 0    0.038076  0.050680  0.061696  0.021872 -0.044223 -0.034821 -0.043401 -0.002592  0.019907 -0.017646

random_state1=1223
random_state2=1223

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_state1)

#2. 모델 구성
model = RandomForestRegressor(random_state=random_state2)

print('random_state :', random_state1, random_state2)

model.fit(x_train, y_train)
print('==========', model.__class__.__name__, '==========')
print('r2 :', model.score(x_test, y_test))
print(model.feature_importances_)

'''
random_state : 1223 1223
========== RandomForestRegressor ==========
r2 : 0.3687286985683689
[0.05394197 0.00931513 0.25953258 0.1125408  0.04297661 0.05293764
 0.06684433 0.02490964 0.29157054 0.08543076]
'''

########## 하위 20~25% 컬럼 제거 ##########

x2 = x
percentiles = np.percentile(model.feature_importances_, 10)

for i, fi in enumerate(model.feature_importances_):
    if fi <= percentiles:
       x2 = x2.drop(datasets.feature_names[i], axis=1)

print(x2)

x_train, x_test, y_train, y_test = train_test_split(
    x2, y, train_size=0.8, random_state=random_state1)

#2. 모델 구성
# model = RandomForestRegressor(random_state=random_state2)

# print('random_state :', random_state1, random_state2)

model.fit(x_train, y_train)
print('==========', model.__class__.__name__, 'DROP ==========')
print('r2 :', model.score(x_test, y_test))
print(model.feature_importances_)

'''
random_state : 1223 1223
========== RandomForestRegressor DROP ==========
r2 : 0.3519026632903155
[0.0559631  0.25932497 0.1145097  0.04406126 0.05386534 0.06725223
 0.02537816 0.29180158 0.08784367]

결과 : 성능 저하

'''
