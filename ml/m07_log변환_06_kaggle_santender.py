import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#1. 데이터
path = "C:/ai5/_data/kaggle/santander-customer-transaction-prediction/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

# x.boxplot()       # 이상치 확인, Population 에 이상치가 있다
# x.plot.box()      # 동일
# plt.show()

# print(x.info())   # 모두 non-null
# print(x.describe)

# df['Population'].boxplot()
# plt.show()        # 시리즈에서 안된다, AttributeError: 'Series' object has no attribute 'boxplot'

# df['Population'].plot.box()
# plt.show()        # 된다

# df['Population'].hist(bins=50)
# plt.show()

# y.hist(bins=50)
# plt.show()          # 0과1

########## Population 로그 변환 ##########
# x['var_45'] = np.log1p(x['var_45']) # 지수변환 np.expm1
# x['var_74'] = np.log1p(x['var_74']) # 지수변환 np.expm1
# x['var_117'] = np.log1p(x['var_117']) # 지수변환 np.expm1
# x['var_120'] = np.log1p(x['var_120']) # 지수변환 np.expm1

# [실습] 만들기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)

# print(x_train)

########## y 로그 변환 ##########
# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)
################################

#2. 모델 구성
model = RandomForestRegressor(random_state=1234, max_depth=5, min_samples_split=3)
# model = LinearRegression()

'''
print(x_train.isnull().any(axis=1))
'''

#3. 훈련
model.fit(x_train, y_train)

#4. 평가,예측
score = model.score(x_test, y_test)     # score = r2_score
print("score : ", score)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)

'''
RandomForestRegressor
로그변환X


y만 로그 변환


val만 로그 변환


로그변환O


##### LinearRegression #####
로그변환X
score :  0.1899892995146354
r2 score :  0.1899892995146354

y만 로그 변환
score :  0.18998929951463528
r2 score :  0.18998929951463528

val만 로그 변환
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

로그변환O
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

'''
