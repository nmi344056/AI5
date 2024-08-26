import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#1. 데이터
datasets = load_boston()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)

df['target'] = datasets.target

# df.boxplot()        # 이상치 확인, Population 에 이상치가 있다
# df.plot.box()       # 동일
# plt.show()

# print(df.info())    # 모두 non-null
# print(df.describe)

# df['Population'].boxplot()
# plt.show()          # 시리즈에서 안된다, AttributeError: 'Series' object has no attribute 'boxplot'

# df['B'].plot.box()
# plt.show()          # 된다

# df['B'].hist(bins=50)
# plt.show()

# df['target'].hist(bins=50)
# plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']

########## Population 로그 변환 ##########
x['B'] = np.log1p(x['B']) # 지수변환 np.expm1

# [실습] 만들기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)

########## y 로그 변환 ##########
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
################################

#2. 모델 구성
model = RandomForestRegressor(random_state=1234, max_depth=5, min_samples_split=3)
# model = LinearRegression()

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
score :  0.8905816885452467
r2 score :  0.8905816885452467

y만 로그 변환
score :  0.8575446276395677
r2 score :  0.8575446276395677

B만 로그 변환
score :  0.8905816885452467
r2 score :  0.8905816885452467

로그변환O
score :  0.8575583103424232
r2 score :  0.8575583103424232

##### LinearRegression #####
로그변환X
score :  0.7665382927362877
r2 score :  0.7665382927362877

y만 로그 변환
score :  0.7948979254138078
r2 score :  0.7948979254138078

B만 로그 변환
score :  0.7710827448613004
r2 score :  0.7710827448613004

로그변환O
score :  0.8004450478272083
r2 score :  0.8004450478272083

'''
