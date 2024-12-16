# 23_1 copy

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, random_state=1223)

#2. 모델 구성
model = RandomForestClassifier(random_state=1223)

model.fit(x_train, y_train)
print('==========', model.__class__.__name__, '==========')
print('acc :', model.score(x_test, y_test))

'''
acc : 0.9777777777777777
'''

print(model.feature_importances_)
'''
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

thresholds = np.sort(model.feature_importances_)    # 오름차순 (1-2-3)
print(thresholds)
'''
[0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 2.55452052e-05 2.82560661e-05 3.24578083e-05 4.56565143e-05
 9.99062839e-05 1.23540702e-04 1.37295796e-04 2.96316379e-04
 5.48899644e-04 6.78704204e-04 1.21113019e-03 1.46684371e-03
 2.03549096e-03 2.39008914e-03 3.28051993e-03 5.48039124e-03
 6.79716398e-03 7.69191223e-03 7.74184369e-03 8.63100042e-03
 8.81822788e-03 9.75321481e-03 1.02211980e-02 1.05410820e-02
 1.08591464e-02 1.23990699e-02 1.33219640e-02 1.72195928e-02
 1.74062193e-02 1.81855748e-02 1.84862656e-02 1.85314325e-02
 2.02555900e-02 2.10193342e-02 2.11484828e-02 2.13446288e-02
 2.14025504e-02 2.14496422e-02 2.15828252e-02 2.20501425e-02
 2.34860951e-02 2.37832862e-02 2.41710385e-02 2.48623827e-02
 2.50370391e-02 2.50731385e-02 2.51334337e-02 2.60401050e-02
 2.63532308e-02 2.71615704e-02 2.85024358e-02 2.85417748e-02
 2.97603141e-02 3.03543272e-02 3.31805152e-02 3.44453029e-02
 3.94845951e-02 4.38164383e-02 4.44882149e-02 5.15856136e-02]
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
Trech=0.000, n=64, ACC: 93.64%
Trech=0.000, n=64, ACC: 93.64%
Trech=0.000, n=64, ACC: 93.64%
Trech=0.000, n=64, ACC: 93.64%
Trech=0.000, n=60, ACC: 94.72%
Trech=0.000, n=59, ACC: 93.43%
Trech=0.000, n=58, ACC: 94.45%
Trech=0.000, n=57, ACC: 92.79%
Trech=0.000, n=56, ACC: 91.77%
Trech=0.000, n=55, ACC: 95.13%
Trech=0.000, n=54, ACC: 94.85%
Trech=0.000, n=53, ACC: 92.76%
Trech=0.001, n=52, ACC: 91.40%
Trech=0.001, n=51, ACC: 91.10%
Trech=0.001, n=50, ACC: 93.33%
Trech=0.001, n=49, ACC: 94.28%
Trech=0.002, n=48, ACC: 94.01%
Trech=0.002, n=47, ACC: 96.01%
Trech=0.003, n=46, ACC: 93.16%
Trech=0.005, n=45, ACC: 93.16%
Trech=0.007, n=44, ACC: 95.67%
Trech=0.008, n=43, ACC: 94.31%
Trech=0.008, n=42, ACC: 94.08%
Trech=0.009, n=41, ACC: 93.16%
Trech=0.009, n=40, ACC: 96.51%  best
Trech=0.010, n=39, ACC: 93.30%
Trech=0.010, n=38, ACC: 95.97%
Trech=0.011, n=37, ACC: 95.26%
Trech=0.011, n=36, ACC: 94.31%
Trech=0.012, n=35, ACC: 96.14%
Trech=0.013, n=34, ACC: 93.20%
Trech=0.017, n=33, ACC: 96.14%
Trech=0.017, n=32, ACC: 91.40%
Trech=0.018, n=31, ACC: 93.43%
Trech=0.018, n=30, ACC: 91.54%
Trech=0.019, n=29, ACC: 94.28%
Trech=0.020, n=28, ACC: 95.40%
Trech=0.021, n=27, ACC: 91.91%
Trech=0.021, n=26, ACC: 95.26%
Trech=0.021, n=25, ACC: 93.23%
Trech=0.021, n=24, ACC: 94.55%
Trech=0.021, n=23, ACC: 92.82%
Trech=0.022, n=22, ACC: 92.65%
Trech=0.022, n=21, ACC: 95.23%
Trech=0.023, n=20, ACC: 95.63%
Trech=0.024, n=19, ACC: 96.24%
Trech=0.024, n=18, ACC: 91.81%
Trech=0.025, n=17, ACC: 91.84%
Trech=0.025, n=16, ACC: 93.97%
Trech=0.025, n=15, ACC: 92.82%
Trech=0.025, n=14, ACC: 87.98%
Trech=0.026, n=13, ACC: 88.69%
Trech=0.026, n=12, ACC: 90.59%
Trech=0.027, n=11, ACC: 86.97%
Trech=0.029, n=10, ACC: 88.19%
Trech=0.029, n=9, ACC: 85.07%
Trech=0.030, n=8, ACC: 83.01%
Trech=0.030, n=7, ACC: 78.20%
Trech=0.033, n=6, ACC: 56.97%
Trech=0.034, n=5, ACC: 39.20%
Trech=0.039, n=4, ACC: 2.06%
Trech=0.044, n=3, ACC: -43.64%
Trech=0.044, n=2, ACC: -76.37%
Trech=0.052, n=1, ACC: -99.59%
'''
