'''
##### 판다스로 바꿔서 컬럼 삭제 #####
pd.DataFrame
컬럼명 : datasets.feature_names 안에 있다.
feature_importance가 전체 중요도에서 하위 20~25% 컬럼들을 PCA
데이터셋 재구성한 후
기존 모델결과와 비교
성능 향상 시키기

'''

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
x3 = x
percentiles = np.percentile(model.feature_importances_, 25)

for i, fi in enumerate(model.feature_importances_):
    if fi <= percentiles:
       x2 = x2.drop(datasets.feature_names[i], axis=1)
    else:
        x3 = x3.drop(datasets.feature_names[i], axis=1)

'''
print(x2)
       MedInc HouseAge  AveRooms  AveOccup Latitude Longitude
0      8.3252     41.0  6.984127  2.555556    37.88   -122.23
print(x3)
      AveBedrms Population
0      1.023810      322.0
'''

x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y, train_size=0.8, random_state=random_state1)

x3_train, x3_test, y3_train, y3_test = train_test_split(
    x3, y, train_size=0.8, random_state=random_state1)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
x3_train = pca.fit_transform(x3_train)
x3_test = pca.transform(x3_test)

print(x2_train.shape, x3_train.shape)   # (16512, 6) (16512, 1)
print(x2_test.shape, x3_test.shape)     # (4128, 6) (4128, 1)

x_train = np.concatenate([x2_train, x3_train], axis=1)
x_test = np.concatenate([x2_test, x3_test], axis=1)

print(x_train.shape)    # (16512, 7)
print(x_test.shape)     # (4128, 7)

model.fit(x_train, y_train)
print('==========', model.__class__.__name__, 'PCA ==========')
print('r2 :', model.score(x_test, y_test))
print(model.feature_importances_)

'''
random_state : 1223 1223
========== XGBRegressor DROP ==========
r2 : 0.8424404298528521
[0.51401776 0.06747776 0.05457867 0.15093587 0.1003058  0.1126841 ]

========== XGBRegressor PCA ==========
r2 : 0.8396699203549337
[0.50174356 0.06651443 0.04998786 0.14618227 0.10080014 0.10979127
 0.02498043]

결과 : DROP과 비교 시 성능 저하

'''
