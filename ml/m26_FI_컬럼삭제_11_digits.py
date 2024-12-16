# 23_1 copy

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# 1. 데이터
datasets = load_digits()      # feature_name 때문에
x = datasets.data
y = datasets.target

x = pd.DataFrame(x, columns=[datasets.feature_names])
print(x.shape)  # (1797, 64)

random_state1=1223
random_state2=1223

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, random_state=random_state1)

#2. 모델 구성
model = RandomForestClassifier(random_state=random_state2)

print('random_state :', random_state1, random_state2)

model.fit(x_train, y_train)
print('==========', model.__class__.__name__, '==========')
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)

'''
random_state : 1223 1223
========== RandomForestClassifier ==========
acc : 0.9777777777777777
[0.00000000e+00 1.46684371e-03 2.15828252e-02 8.63100042e-03
 7.74184369e-03 2.11484828e-02 9.75321481e-03 6.78704204e-04
 9.99062839e-05 1.05410820e-02 2.10193342e-02 7.69191223e-03
 1.81855748e-02 2.60401050e-02 5.48039124e-03 5.48899644e-04
 3.24578083e-05 6.79716398e-03 2.34860951e-02 2.37832862e-02
 3.03543272e-02 5.15856136e-02 8.81822788e-03 2.96316379e-04
 2.55452052e-05 1.23990699e-02 4.38164383e-02 2.48623827e-02
 3.31805152e-02 2.20501425e-02 2.71615704e-02 4.56565143e-05
 0.00000000e+00 2.85024358e-02 2.85417748e-02 1.85314325e-02
 3.94845951e-02 2.02555900e-02 2.41710385e-02 0.00000000e+00
 2.82560661e-05 1.08591464e-02 3.44453029e-02 4.44882149e-02
 2.14025504e-02 1.74062193e-02 2.14496422e-02 1.37295796e-04
 1.23540702e-04 2.39008914e-03 1.72195928e-02 2.13446288e-02
 1.33219640e-02 2.63532308e-02 2.50370391e-02 1.21113019e-03
 0.00000000e+00 2.03549096e-03 2.50731385e-02 1.02211980e-02
 2.51334337e-02 2.97603141e-02 1.84862656e-02 3.28051993e-03]
'''

########## 하위 20~25% 컬럼 제거 ##########

x2 = x
percentiles = np.percentile(model.feature_importances_, 45)

for i, fi in enumerate(model.feature_importances_):
    if fi <= percentiles:
       x2 = x2.drop(datasets.feature_names[i], axis=1)

print(x2.shape)  # (1797, 35)

x_train, x_test, y_train, y_test = train_test_split(
    x2, y, train_size=0.8, stratify=y, random_state=random_state1)

# model = RandomForestClassifier(random_state=random_state2)

# print('random_state :', random_state1, random_state2)

model.fit(x_train, y_train)
print('==========', model.__class__.__name__, 'DROP ==========')
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)

'''
========== RandomForestClassifier DROP ==========
acc : 0.9833333333333333
[0.02734288 0.02472079 0.03020975 0.01968706 0.02863839 0.02631697
 0.0268591  0.03087187 0.05623924 0.01660876 0.04512588 0.03386319
 0.03727147 0.02616843 0.03217584 0.03159522 0.02747815 0.01735251
 0.03988159 0.02046969 0.02692841 0.04200432 0.04701271 0.02453198
 0.02161957 0.02554952 0.018218   0.02390689 0.01514343 0.03145336
 0.02870197 0.02606011 0.02763365 0.02617448 0.01618484]

결과 : 성능 향상

'''
