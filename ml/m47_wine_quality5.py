import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# [실습] y는 quality로 만들기

#1. 데이터
path = 'C:\\ai5\\_data\\kaggle\\wine\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

print(train_csv['quality'].value_counts().sort_index())
# 3      26
# 4     186
# 5    1788
# 6    2416
# 7     924
# 8     152
# 9       5

le = LabelEncoder()
le.fit(train_csv['type'])
aaa = le.transform(train_csv['type'])
print(aaa)          # [1 0 1 ... 1 1 1]
print(type(aaa))    # <class 'numpy.ndarray'>
print(aaa.shape)    # (5497,)
train_csv['type'] = aaa

print(le.transform(['red', 'white']))       # [0 1]

print(train_csv.describe())
print(train_csv.info())

x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']
print(x, x.shape)   # (5497, 12)
print(y, y.shape)   # (5497,)

# y = y - 3           # LabelEncoder 사용 X

############################################################
# [실습] y의 클래스가 7개에서 3~5개로 줄어들 경우 성능을 비교
# hint : for문
############################################################
y = y.copy()

for i, v in enumerate(y):
    if v <= 4:
        y[i] = 0
    elif v <= 7:
        y[i] = 1
    else:
        y[i] = 2

print(y.value_counts().sort_index())
# 0     212
# 1    5128
# 2     157

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=123,
    stratify=y,
    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성
parameters = {
    'n_estimators' : 100,
    'learning_rate' : 0.1,
    'max_depth' : 5,
}

model = XGBClassifier(**parameters, n_jobs=-1)
model.set_params(early_stopping_rounds=500,
                 eval_metric='merror'
                 )

#3. 훈련
model.fit(x_train, y_train,
          eval_set=[(x_test, y_test)],
          verbose=1)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('model.score :', results)

y_predict = model.predict(x_test)
print('acc :', accuracy_score(y_test, y_predict))
print('f1 :', f1_score(y_test, y_predict, average='macro'))

'''
y 클래스가 7개
model.score : 0.6145454545454545
acc : 0.6145454545454545
f1 : 0.28412757419305473

y 클래스가 3개
3 4 / 5 6 7 / 8 9
model.score : 0.9418181818181818
acc : 0.9418181818181818
f1 : 0.5016044526326855

'''
