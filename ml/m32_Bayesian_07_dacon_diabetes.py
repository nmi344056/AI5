# https://dacon.io/competitions/official/236068/mysubmission?isSample=1

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import xgboost as xgb
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

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
print(x)                    # [652 rows x 8 columns]
y = train_csv['Outcome']
print(y.shape)              # (652,)

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

# n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

early_stop = xgb.callback.EarlyStopping(
    rounds=50,                      # patience
    # metric_name='logloss',        # 이진: logloss, 다중: mlogloss
    data_name='validation_0',
    # save_best=True,               # AttributeError: `best_iteration` is only defined when early stopping is used.
)

#2. 모델 구성
bayesian_params={
    'learning_rate' : (0.001, 0.1),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (9, 500),
    'reg_lambda' : (-0.001, 10),
    'reg_alpha' : (0.01, 50),
}

def xgb_hamsu(learning_rate, max_depth, 
              num_leaves, min_child_samples, min_child_weight, 
              subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha,
              ):
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),        # 무조건 정수형
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample, 1), 0),    # 0 ~ 1 사이의 값
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)), 10),   # 무조건 10 이상
        'reg_lambda' : max(reg_lambda, 0),          # 무조건 양수만
        'reg_alpha' : reg_alpha,
    }

    model = XGBClassifier(**params,
                        #  n_jobs=-1,
                         tree_mothod='gpu_hist',
                         gpu_id=0,
                         callbacks=[early_stop],
                         )

    model.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            # eval_metric='logloss',
            verbose=0,
            )

    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    return results

bay = BayesianOptimization(                 # 모델 정의
    f=xgb_hamsu,
    pbounds=bayesian_params,
    random_state=333,
)

n_iter = 100
start = time.time()
bay.maximize(init_points=5, n_iter=n_iter)  # fit
end = time.time()

print(bay.max)
print(n_iter, '번 걸린시간 :', round(end - start, 2), '초')

'''
{'target': 0.7404580152671756, 'params': {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_bin': 34.61659196195003, 'max_depth': 6.441464628853241, 'min_child_samples': 47.7283022819055, 'min_child_weight': 1.0, 'num_leaves': 30.573338980101244, 'reg_alpha': 
5.863915785619019, 'reg_lambda': -0.001, 'subsample': 1.0}}
100 번 걸린시간 : 18.7 초

| 77        | 0.7328    | 0.5       | 0.1       | 31.5      | 3.0       | 51.57     | 1.0       | 31.54     | 2.54      | -0.001    | 1.0       |
| 78        | 0.5954    | 0.8739    | 0.00375   | 10.49     | 3.774     | 69.45     | 8.458     | 24.8      | 3.075     | 4.236     | 0.5021    |
| 79        | 0.7099    | 0.5       | 0.1       | 28.04     | 3.0       | 54.56     | 1.0       | 30.93     | 8.408     | -0.001    | 0.5       |
| 80        | 0.7405    | 0.5       | 0.1       | 34.62     | 6.441     | 47.73     | 1.0       | 30.57     | 5.864     | -0.001    | 1.0       |
'''
