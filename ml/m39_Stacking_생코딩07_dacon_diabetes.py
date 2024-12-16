# https://dacon.io/competitions/official/236068/mysubmission?isSample=1

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingRegressor, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

#1. 데이터
path = "C:\\ai5\\_data\\dacon\\diabetes\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.columns)
# Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# print(train_csv.info())     # 결측치가 없다
# print(test_csv.info())      # 결측치가 없다
# print(train_csv.isnull().sum())
# print(test_csv.isnull().sum())

x = train_csv.drop(['Outcome'], axis=1)
# print(x)                    # [652 rows x 8 columns]
y = train_csv['Outcome']
# print(y.shape)              # (652,)

# print(np.unique(y, return_counts=True))     
# (array([0, 1], dtype=int64), array([424, 228], dtype=int64))
# print(pd.DataFrame(y).value_counts())
# 0          424
# 1          228

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123)

scaler = MaxAbsScaler()
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
XGBClassifier ACC : 0.6947
RandomForestClassifier ACC : 0.7176
CatBoostClassifier ACC : 0.7252
'''

x_train_new = np.array(train_list).T
print(x_train_new.shape)    # (521, 3)

x_test_new = np.array(test_list).T
print(x_test_new.shape)     # (131, 3)

#2. 모델
model2 = CatBoostClassifier(verbose=0)
model2.fit(x_train_new, y_train)
y_pred = model2.predict(x_test_new)
score2 = accuracy_score(y_test, y_pred)
print('스태킹 결과 :', score2)

'''
스태킹 결과 : 0.7099236641221374
'''
