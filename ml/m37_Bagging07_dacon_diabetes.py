# https://dacon.io/competitions/official/236068/mysubmission?isSample=1

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

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
# model = DecisionTreeClassifier()

# model = BaggingClassifier(DecisionTreeClassifier(),
#                           n_estimators=100,
#                           n_jobs=-1,
#                           random_state=4444,
#                           bootstrap=True,   # 중복 허용, Default
#                         #   bootstrap=False,  # 중복 허용 안함
#                           )

# model = LogisticRegression()

# model = RandomForestClassifier()

model = BaggingClassifier(LogisticRegression(),
                          n_estimators=100,
                          n_jobs=-1,
                          random_state=4444,
                          bootstrap=True,   # 중복 허용, Default
                        #   bootstrap=False,  # 중복 허용 안함
                          )

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 :', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score: ', acc)

'''
DecisionTreeRegressor()
최종점수 : 0.648854961832061
acc_score:  0.648854961832061

BaggingRegressor(DecisionTreeRegressor(), bootstrap=True
최종점수 : 0.7175572519083969
acc_score:  0.7175572519083969

BaggingRegressor(DecisionTreeRegressor(), bootstrap=False
최종점수 : 0.6717557251908397
acc_score:  0.6717557251908397

LogisticRegression()
최종점수 : 0.7175572519083969
acc_score:  0.7175572519083969

RandomForestRegressor()
최종점수 : 0.7099236641221374
acc_score:  0.7099236641221374

BaggingRegressor(RandomForestRegressor(), bootstrap=True
최종점수 : 0.7175572519083969
acc_score:  0.7175572519083969

BaggingRegressor(RandomForestRegressor(), bootstrap=False
최종점수 : 0.7175572519083969
acc_score:  0.7175572519083969

'''
