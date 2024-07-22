# https://dacon.io/competitions/official/236068/data
# 풀어라!!!

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
path = 'C:/AI5/_data/dacon/diabetes/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)    #[652 rows x 9 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)     # [116 rows x 8 columns]

submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
print(submission_csv)   # [116 rows x 1 columns]

print(train_csv.shape)  # (652, 9)
print(test_csv.shape)   # (116, 8)
print(submission_csv.shape) # (116, 1)

print(train_csv.columns)

train_csv.info()    # 결측치 없음
test_csv.info()     # 노 프라블름

# train_csv = train_csv[train_csv['BloodPressure'] > 0]
# train_csv = train_csv[train_csv['BMI'] > 0.0]

x = train_csv.drop(['Outcome'], axis=1)
print(x)    # [652 rows x 8 columns]
y = train_csv['Outcome']
print(y.shape)  # (652,)

print(np.unique(y, return_counts=True))
print(type(x))      # (array([0, 1], dtype=int64), array([424, 228], dtype=int64))

print(pd.DataFrame(y).value_counts())
# 0      424
# 1      228
pd.value_counts(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=3333)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=8))

model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 64,
    restore_best_weights=True
)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
                 validation_split=0.3,
                 callbacks=[es]
                 )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('로스 :', loss)
print("acc :", round(loss[1],3))

y_pred = model.predict(x_test)
print(y_pred[:50])
y_pred = np.round(y_pred)
print(y_pred[:50])

acc_score = accuracy_score(y_test, y_pred)

print('acc_score :', acc_score)
print('걸린시간 : ', round(end - start, 2), "초")

y_submit = np.round(model.predict(test_csv))      # round 꼭 넣기
print(y_submit)
print(y_submit.shape)     

#################  submission.csv 만들기 // count 컬럼에 값만 넣어주면 된다 ######
submission_csv['Outcome'] = y_submit
print(submission_csv)
print(submission_csv.shape)

submission_csv.to_csv(path + "submission_0722.csv")

print('로스 :', loss)
print("acc :", round(loss[1],3))




# x,y,train_size=0.8, random_state=415,x_train, y_train, epochs=500, 
# batch_size=25,  validation_split=0.2, patience=24
# 로스 : [0.15757298469543457, 0.7480915784835815]
# acc : 0.748

# x,y,train_size=0.8, # random_state=1)
# 로스 : [0.16639672219753265, 0.7938931584358215]
# acc : 0.794



