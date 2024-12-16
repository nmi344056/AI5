# Pdeudo Labeling 기법 : 모델을 돌려서 나온 결과로 결측지를 찾는다.
# 스태킹 : 모델을 돌려서 나온 결과로 걸럼을 구성해서 새로운 데이터셋을 만든다.
        # 한 데이터로 여러 모델을 돌려서 돌리는 족족 컬럼 만들기

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingRegressor, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, shuffle=True, random_state=1186)

print(x_train.shape, x_test.shape)  # (455, 30) (114, 30)
print(y_train.shape, y_test.shape)  # (455,) (114,)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
xgb = XGBClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier(verbose=0)

train_list = []
test_list = []
models = [xgb, rf, cat]

for model in models:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_train)
    y_test_predict = model.predict(x_test)

    train_list.append(y_predict)
    test_list.append(y_test_predict)

    score = accuracy_score(y_test, y_test_predict)
    class_name = model.__class__.__name__
    print('{0} ACC : {1:.4f}'.format(class_name, score))

print('넘파이 :', np.__version__)   # 넘파이 : 1.22.4 (1.19.5)

'''
XGBClassifier ACC : 0.9912
RandomForestClassifier ACC : 0.9649
CatBoostClassifier ACC : 0.9912
'''

x_train_new = np.array(train_list).T
print(x_train_new.shape)    # (455, 3)

x_test_new = np.array(test_list).T
print(x_test_new.shape)     # (114, 3)

#2. 모델
model2 = CatBoostClassifier(verbose=0)
model2.fit(x_train_new, y_train)
y_pred = model2.predict(x_test_new)
score2 = accuracy_score(y_test, y_pred)
print('스태킹 결과 :', score2)

'''
스태킹 결과 : 0.9912280701754386
'''
