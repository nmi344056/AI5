# https://dacon.io/competitions/official/236068/mysubmission?isSample=1

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR

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

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. 모델 구성
model = SVR()

#3. 훈련
start = time.time()
scores = cross_val_score(model, x, y, cv=kfold)
end = time.time()

print('acc :', scores, 'avg acc :', round(np.mean(scores), 4))
print("time : ", round(end - start, 2), "초")

'''
loss :  0.13780251145362854
accuracy :  0.786

SVR
acc : [0.15482023 0.22418508 0.23024554 0.35424312 0.20443268] avg acc : 0.2336
time :  0.04 초

'''
