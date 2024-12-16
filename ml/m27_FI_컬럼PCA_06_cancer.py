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
x3 = x
percentiles = np.percentile(model.feature_importances_, 25)

for i, fi in enumerate(model.feature_importances_):
    if fi <= percentiles:
       x2 = x2.drop(datasets.feature_names[i], axis=1)
    else:
        x3 = x3.drop(datasets.feature_names[i], axis=1)

print(x2.shape, x3.shape)  # (569, 10) (569, 20)

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
========== DecisionTreeClassifier DROP ==========
acc : 0.9473684210526315
[0.05030732 0.         0.03023319 0.00883871 0.74977167 0.
 0.02648232 0.00907558 0.12529121 0.        ]

========== DecisionTreeClassifier PCA ==========
r2 : 0.5789473684210527
[0.14283857 0.1503043  0.13319559 0.08910569 0.05218434 0.04416534
 0.04213702 0.10636962 0.08511535 0.11021241 0.04437177]

결과 : DROP과 비교 시 성능 저하

'''
