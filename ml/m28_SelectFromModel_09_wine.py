# 23_1 copy

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_wine(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, random_state=1223)

#2. 모델 구성
model = RandomForestClassifier(random_state=1223)

model.fit(x_train, y_train)
print('==========', model.__class__.__name__, '==========')
print('acc :', model.score(x_test, y_test))

'''
acc : 0.9444444444444444
'''

print(model.feature_importances_)
'''
[0.13789135 0.02251876 0.01336314 0.03826336 0.02830375 0.05255915
 0.14261827 0.00916645 0.03234439 0.13563367 0.07199803 0.13963923
 0.17570046]
'''

thresholds = np.sort(model.feature_importances_)    # 오름차순 (1-2-3)
print(thresholds)
'''
[0.00916645 0.01336314 0.02251876 0.02830375 0.03234439 0.03826336
 0.05255915 0.07199803 0.13563367 0.13789135 0.13963923 0.14261827
 0.17570046]
'''

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    select_model = RandomForestClassifier(random_state=1223)

    select_model.fit(select_x_train, y_train)

    select_y_predict = select_model.predict(select_x_test)
    score = r2_score(y_test, select_y_predict)

    print('Trech=%.3f, n=%d, ACC: %.2f%%' %(i, select_x_train.shape[1], score*100))

'''
Trech=0.009, n=13, ACC: 90.86%
Trech=0.013, n=12, ACC: 90.86%
Trech=0.023, n=11, ACC: 90.86%
Trech=0.028, n=10, ACC: 90.86%
Trech=0.032, n=9, ACC: 90.86%
Trech=0.038, n=8, ACC: 95.43%   best
Trech=0.053, n=7, ACC: 90.86%
Trech=0.072, n=6, ACC: 90.86%
Trech=0.136, n=5, ACC: 86.29%
Trech=0.138, n=4, ACC: 86.29%
Trech=0.140, n=3, ACC: 81.73%
Trech=0.143, n=2, ACC: 68.02%
Trech=0.176, n=1, ACC: 22.34%
'''
