import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=1223)

#2. 모델 구성
model = RandomForestRegressor(random_state=1223)

model.fit(x_train, y_train)
print('==========', model.__class__.__name__, '==========')
print('r2 :', model.score(x_test, y_test))

'''
r2 : 0.3687286985683689
'''

print(model.feature_importances_)
'''
[0.05394197 0.00931513 0.25953258 0.1125408  0.04297661 0.05293764
 0.06684433 0.02490964 0.29157054 0.08543076]
'''

thresholds = np.sort(model.feature_importances_)    # 오름차순 (1-2-3)
print(thresholds)
'''
[0.00931513 0.02490964 0.04297661 0.05293764 0.05394197 0.06684433
 0.08543076 0.1125408  0.25953258 0.29157054]
'''

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    select_model = RandomForestRegressor(random_state=1223)

    select_model.fit(select_x_train, y_train)

    select_y_predict = select_model.predict(select_x_test)
    score = r2_score(y_test, select_y_predict)

    print('Trech=%.3f, n=%d, ACC: %.2f%%' %(i, select_x_train.shape[1], score*100))

'''
Trech=0.009, n=10, ACC: 36.87%  best
Trech=0.025, n=9, ACC: 35.19%
Trech=0.043, n=8, ACC: 34.16%
Trech=0.053, n=7, ACC: 33.41%
Trech=0.054, n=6, ACC: 31.90%
Trech=0.067, n=5, ACC: 32.13%
Trech=0.085, n=4, ACC: 30.04%
Trech=0.113, n=3, ACC: 30.48%
Trech=0.260, n=2, ACC: 33.24%
Trech=0.292, n=1, ACC: -5.46%
'''
