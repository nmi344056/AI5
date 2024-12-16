import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

np.random.seed(123)

# 다층 분류 데이터 생성 함수
def create_multiclass_data_with_labels():
    # X 데이터 생성 (20, 3)
    X = np.random.rand(20, 3)

    # y 데이터 생성 (20, 3)
    y = np.random.randint(0, 5, size=(20, 3))

    # 데이터프레임으로 변환
    X_df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3'])
    y_df = pd.DataFrame(y, columns=['Label1', 'Label2', 'Label3'])

    return X_df, y_df
    # return X_df.values, y_df.values

X, y = create_multiclass_data_with_labels()
print('X 데이터 :')
print(X)
print('\nY 데이터 :')
print(y)

print(X.shape)      # (20, 3)
print(y.shape)      # (20, 3)

# # 2. 모델
model = RandomForestClassifier()
model.fit(X, y)
y_pred = model.predict(X)
print(model.__class__.__name__, '스코어 :',
      round(mean_absolute_error(y, y_pred), 4))           # RandomForestClassifier 스코어 : 0.0
print(model.predict([[0.663621, 0.644600, 0.041906]]))    # [[2 0 1]]

# model = LinearRegression()                                # LinearClassifier 없다?
# model.fit(X, y)
# y_pred = model.predict(X)
# print(model.__class__.__name__, '스코어 :',
#       round(mean_absolute_error(y, y_pred), 4))           # LinearRegression 스코어 : 1.0429
# print(model.predict([[0.663621, 0.644600, 0.041906]]))    # [[3.40223864 1.98666602 2.04856613]]

# model = Ridge()
# model.fit(X, y)
# y_pred = model.predict(X)
# print(model.__class__.__name__, '스코어 :',
#       round(mean_absolute_error(y, y_pred), 4))           # Ridge 스코어 : 1.1517
# print(model.predict([[0.663621, 0.644600, 0.041906]]))    # [[2.32191145 2.06324977 2.18675183]]

# ########## ########## ########## ########## ##########

from sklearn.multioutput import MultiOutputClassifier, MultiOutputClassifier

# # model = XGBClassifier()                                 # multioutput is not supported by the current objective function
# model = MultiOutputClassifier(XGBClassifier())
# model.fit(X, y)
# y_pred = model.predict(X)
# print(model.__class__.__name__, '스코어 :',
#       round(mean_absolute_error(y, y_pred), 4))           # MultiOutputClassifier 스코어 : 0.0
# print(model.predict([[0.663621, 0.644600, 0.041906]]))    # [[2 3 3]]

# # model = CatBoostClassifier()                            # TypeError: unhashable type: 'numpy.ndarray'
# model = MultiOutputClassifier(CatBoostClassifier())
# model.fit(X, y)
# # y_pred = model.predict(X)                               # ValueError: Found input variables with inconsistent numbers of samples: [20, 1]
# y_pred = model.predict(X).reshape(-1, 3)
# # print(X.shape)          # (20, 3)
# # print(y_pred.shape)     # (1, 20, 3) -> (20, 3)
# print(model.__class__.__name__, '스코어 :',
#       round(mean_absolute_error(y, y_pred), 4))           # MultiOutputClassifier 스코어 : 0.0
# print(model.predict([[0.663621, 0.644600, 0.041906]]))    # [[[4 3 0]]]

# # model = LGBMClassifier()                                # ValueError: y should be a 1d array, got an array of shape (20, 3) instead.
# model = MultiOutputClassifier(LGBMClassifier())
# model.fit(X, y)
# y_pred = model.predict(X)
# print(model.__class__.__name__, '스코어 :',
#       round(mean_absolute_error(y, y_pred), 4))           # MultiOutputClassifier 스코어 : 1.35
# print(model.predict([[0.663621, 0.644600, 0.041906]]))    # [0 4 3]]

# ########## ########## ########## ########## ##########
# model = CatBoostClassifier(loss_function='MultiRMSE')     # _catboost.CatBoostError: Invalid loss_function='MultiRMSE': for classifier use Logloss, CrossEntropy, MultiClass, MultiClassOneVsAll or custom objective object
# model.fit(X, y)
# y_pred = model.predict(X)
# print(model.__class__.__name__, '스코어 :',
#       round(mean_absolute_error(y, y_pred), 4))
# print(model.predict([[0.663621, 0.644600, 0.041906]]))
