import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

pf = PolynomialFeatures(degree=2, include_bias=False)
x_poly = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_poly, y, train_size=0.8, random_state=1223)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
xgb = XGBRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor(verbose=0)

model = StackingRegressor(
    estimators=[('XGB', xgb), ('RF', rf), ('CAT', cat)],
    final_estimator=CatBoostRegressor(verbose=0),
    n_jobs=-1,
    cv=5,
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score :', model.score(x_test, y_test))
print('r2_score :', r2_score(y_test, y_pred))

'''
model.score : 0.8502735639207767
r2_score : 0.8502735639207767
'''
