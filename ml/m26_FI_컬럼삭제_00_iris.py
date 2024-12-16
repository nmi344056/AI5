'''
##### 판다스로 바꿔서 컬럼 삭제 #####
pd.DataFrame
컬럼명 : datasets.feature_names 안에 있다.
feature_importance가 전체 중요도에서 하위 20~25% 컬럼들을 .drop으로 제거해서
데이터셋 재구성한 후
기존 모델결과와 비교
성능 향상 시키기

'''
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# 1. 데이터
datasets = load_iris()      # feature_name 때문에
x = datasets.data
y = datasets.target

x = pd.DataFrame(x, columns=[datasets.feature_names])
# print(x)
#     sepal length (cm) sepal width (cm) petal length (cm) petal width (cm)
# 0                 5.1              3.5               1.4              0.2

random_state1=123
random_state2=7777

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
random_state : 123 7777
========== DecisionTreeClassifier ==========
acc : 0.8333333333333334
[0.         0.0425     0.42133357 0.53616643]
'''

########## 하위 20~25% 컬럼 제거 ##########

x2 = x
percentiles = np.percentile(model.feature_importances_, 50)

for i, fi in enumerate(model.feature_importances_):
    if fi <= percentiles:
       x2 = x2.drop(datasets.feature_names[i], axis=1)

# print(x2)
#     sepal width (cm) petal length (cm) petal width (cm)
# 0                3.5               1.4              0.2

x_train, x_test, y_train, y_test = train_test_split(
    x2, y, train_size=0.8, stratify=y, random_state=random_state1)

#2. 모델 구성
model = DecisionTreeClassifier(random_state=random_state2)

print('random_state :', random_state1, random_state2)

model.fit(x_train, y_train)
print('==========', model.__class__.__name__, 'DROP ==========')
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)

'''
random_state : 123 7777
========== DecisionTreeClassifier DROP ==========
acc : 0.9333333333333333
[0.44565425 0.55434575]

결과 : 성능 향상

'''
