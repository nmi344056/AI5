import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=1223)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
# model = DecisionTreeRegressor()

# model = BaggingRegressor(DecisionTreeRegressor(),
#                           n_estimators=100,
#                           n_jobs=-1,
#                           random_state=4444,
#                           bootstrap=True,   # 중복 허용, Default
#                         #   bootstrap=False,  # 중복 허용 안함
#                           )

# model = RandomForestRegressor()

model = BaggingRegressor(RandomForestRegressor(),
                          n_estimators=100,
                          n_jobs=-1,
                          random_state=4444,
                        #   bootstrap=True,   # 중복 허용, Default
                          bootstrap=False,  # 중복 허용 안함
                          )

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 :', results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2_score: ', r2)

'''
DecisionTreeRegressor()
최종점수 : 0.5941748294235323
r2_score:  0.5941748294235323

BaggingRegressor(DecisionTreeRegressor(), bootstrap=True
최종점수 : 0.8112162827563815
r2_score:  0.8112162827563815

BaggingRegressor(DecisionTreeRegressor(), bootstrap=False
최종점수 : 0.6391308951670642
r2_score:  0.6391308951670642

RandomForestRegressor()
최종점수 : 0.812549051773114
r2_score:  0.812549051773114

BaggingRegressor(RandomForestRegressor(), bootstrap=True
최종점수 : 0.8066286540747862
r2_score:  0.8066286540747862

BaggingRegressor(RandomForestRegressor(), bootstrap=False   *
최종점수 : 0.8146178224975462
r2_score:  0.8146178224975462

'''
