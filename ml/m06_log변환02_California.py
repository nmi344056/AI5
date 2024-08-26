import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#1. 데이터
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)

df['target'] = datasets.target

df.boxplot()        # 이상치 확인, Population 에 이상치가 있다
# df.plot.box()     # 동일
plt.show()

print(df.info())    # 모두 non-null
print(df.describe)

# df['Population'].boxplot()
# plt.show()        # 시리즈에서 안된다, AttributeError: 'Series' object has no attribute 'boxplot'

# df['Population'].plot.box()
# plt.show()        # 된다

# df['Population'].hist(bins=50)
# plt.show()

# df['target'].hist(bins=50)
# plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']

########## Population 로그 변환 ##########
x['Population'] = np.log1p(x['Population']) # 지수변환 np.expm1

# [실습] 만들기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)

########## y 로그 변환 ##########
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
################################

#2. 모델 구성
# model = RandomForestRegressor(random_state=1234, max_depth=5, min_samples_split=3)
model = LinearRegression()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가,예측
score = model.score(x_test, y_test)     # score = r2_score
print("score : ", score)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)

'''
RandomForestRegressor       # 트리계열의 구조는 이상치,결측치에 자유롭다., 이상치,결측치 처리를 안해도 된다.
로그변환X
score :  0.6495152533878351
r2 score :  0.6495152533878351

y만 로그 변환
score :  0.6584197269397019
r2 score :  0.6584197269397019

Population만 로그 변환
score :  0.6495031475648194
r2 score :  0.6495031475648194

로그변환O
score :  0.6584197269397019
r2 score :  0.6584197269397019

##### LinearRegression #####
로그변환X
score :  0.606572212210644
r2 score :  0.606572212210644

y만 로그 변환
score :  0.6295290651919585
r2 score :  0.6295290651919585

Population만 로그 변환
score :  0.606598836886877
r2 score :  0.606598836886877

로그변환O
score :  0.6294707351612604
r2 score :  0.6294707351612604

'''
