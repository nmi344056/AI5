import numpy as np
import pandas as pd
import time
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_covtype

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (581012, 54) (581012,)

print(pd.value_counts(y))    
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747
# 문제 : 0이 없다, onehot을 0이 아닌 1부터 시작한다.

print(y)
print(np.unique(y, return_counts=True))

# from tensorflow.keras.utils import to_categorical
# y_ohe = to_categorical(y)
# print(y_ohe)
# print(y_ohe.shape)              # (581012, 8)

y_ohe = pd.get_dummies(y)          # pandas
print(y_ohe)                       # 1  2  3  4  5  6  7
print(y_ohe.shape)                 # (581012, 7)

# print("==============================")
# from sklearn.preprocessing import OneHotEncoder
# y_ohe = y.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)        # True가 default
# ohe.fit(y_ohe)
# y_ohe = ohe.transform(y_ohe)
# print(y_ohe)
# print(y_ohe.shape)                 # (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, train_size=0.9, random_state=6666,
                                                    stratify=y)

# print(pd.value_counts(y_train))
# 2    141429       141651
# 1    106155       105920
# 3     17958       17877
# 7     10262       10255
# 6      8613       8683
# 5      4711       4747
# 4      1378       1373

# print(x_train.shape, x_test.shape)      # (522910, 54) (58102, 54)
# print(y_train.shape, y_test.shape)      # (522910, 8) (58102, 8)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train)
# print(np.min(x_train), np.max(x_train))     # 0.0 1.0
# print(np.min(x_test), np.max(x_test))       # -0.009150326797385616 1.0026143790849673

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'tree_method': ['gpu_hist'], 'n_jobs':[-1], 'n_estimators':[100, 500], 'max_depth':[6, 10, 12], 'min_samples_leaf':[3, 10]},    # 2*3*2=12번
    {'tree_method': ['gpu_hist'], 'n_jobs':[-1], 'max_depth':[6, 8, 10, 12], 'min_samples_leaf':[3, 5, 7, 10]},                      # 4*4=16번
    {'tree_method': ['gpu_hist'], 'n_jobs':[-1], 'min_samples_leaf':[2, 3, 5, 10], 'min_samples_split':[2, 3, 5, 10]},               # 4*4=16번
    {'tree_method': ['gpu_hist'], 'n_jobs':[-1], 'min_samples_leaf':[2, 3, 5, 10]},                                                  # 4=4번
]   # 12+16+16+4=48

#2. 모델 구성
model = GridSearchCV(xgb.XGBClassifier(), parameters, cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1,)

start = time.time()
model.fit(x_train, y_train)
end = time.time()

print('최적의 파라미터 :', model.best_params_)
print('model.score :', model.score(x_test, y_test))

y_predict_best = model.best_estimator_.predict(x_test)
print('최적의 튠 acc :', accuracy_score(y_test, y_predict_best))

print("time : ", round(end - start, 2), "초")

'''
128 256 256 256 126 8 / train_size=0.9, random_state=6666 / epochs=100, batch_size=300, validation_split=0.2

loss :  0.28132835030555725
accuracy :  0.888

GridSearchCV


'''
