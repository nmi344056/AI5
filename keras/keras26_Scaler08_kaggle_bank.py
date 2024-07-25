# keras21_2_kaggle_bank copy

# https://www.kaggle.com/competitions/playground-series-s4e1/data?select=train.csv

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

# #0 replace data
# PATH = 'C:/AI5/_data/kaglle/playground-series-s4e1/'

# train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

# print(train_csv['Geography'].value_counts())

# train_csv['Geography'] = train_csv['Geography'].replace('France', value = 1)
# train_csv['Geography'] = train_csv['Geography'].replace('Spain', value = 2)
# train_csv['Geography'] = train_csv['Geography'].replace('Germany', value = 3)

# train_csv['Gender'] = train_csv['Gender'].replace('Male', value = 1)
# train_csv['Gender'] = train_csv['Gender'].replace('Female', value = 2)

# train_csv.to_csv(PATH + "replaced_train.csv")

# #1 replace data
# PATH = 'C:/AI5/_data/kaglle/playground-series-s4e1/'

# test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)

# print(test_csv['Geography'].value_counts())

# test_csv['Geography'] = test_csv['Geography'].replace('France', value = 1)
# test_csv['Geography'] = test_csv['Geography'].replace('Spain', value = 2)
# test_csv['Geography'] = test_csv['Geography'].replace('Germany', value = 3)

# test_csv['Gender'] = test_csv['Gender'].replace('Male', value = 1)
# test_csv['Gender'] = test_csv['Gender'].replace('Female', value = 2)

# test_csv.to_csv(PATH + "replaced_test.csv")



#1. 데이터
path = 'C:/AI5/_data/kaglle/playground-series-s4e1/'

train_csv = pd.read_csv(path + 'replaced_train.csv', index_col=0)
print(train_csv)    # [165034 rows x 13 columns]

test_csv = pd.read_csv(path + 'replaced_test.csv', index_col=0)
print(test_csv)     # [110023 rows x 12 columns]

submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
print(submission_csv)      # [110023 rows x 1 columns]

print(train_csv.shape)      # (165034, 13)
print(test_csv.shape)       # (110023, 12)
print(submission_csv.shape) # (110023, 1)

print(train_csv.columns)

train_csv.info()    # 결측치 없음
test_csv.info()     # 결측치 없음

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

###############################################
train_csv=train_csv.drop(['CustomerId', 'Surname'], axis=1)

from sklearn.preprocessing import MinMaxScaler

train_scaler = MinMaxScaler()

train_csv_copy = train_csv.copy()

train_csv_copy = train_csv_copy.drop(['Exited'], axis = 1)

train_scaler.fit(train_csv_copy)

train_csv_scaled = train_scaler.transform(train_csv_copy)

train_csv = pd.concat([pd.DataFrame(data = train_csv_scaled), train_csv['Exited']], axis = 1)

test_scaler = MinMaxScaler()

test_csv_copy = test_csv.copy()

test_scaler.fit(test_csv_copy)

test_csv_scaled = test_scaler.transform(test_csv_copy)

test_csv = pd.DataFrame(data = test_csv_scaled)
###############################################

x = train_csv.drop(['Exited'], axis = 1)
print(x)    # [165034 rows x 10 columns]

y = train_csv['Exited']
print(y.shape)      # (165034,)


print(np.unique(y, return_counts=True))
print(type(x))      # (array([0, 1], dtype=int64), array([424, 228], dtype=int64))

print(pd.DataFrame(y).value_counts())
# 0      424
# 1      228
pd.value_counts(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=1186)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print('x_train :', x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#2. 모델구성
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=10))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 40,
    restore_best_weights=True
)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=333,
                 validation_split=0.2,
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

print(test_csv.shape)

y_submit = np.round(model.predict(test_csv))      # round 꼭 넣기
print(y_submit)
print(y_submit.shape)     

#################  submission.csv 만들기 // count 컬럼에 값만 넣어주면 된다 ######
submission_csv['Exited'] = y_submit
print(submission_csv)
print(submission_csv.shape)

submission_csv.to_csv(path + "submission_0723_1223.csv")

print('로스 :', loss)
print("acc :", round(loss[1],3))

# x,y,train_size=0.8, random_state=1186
# 로스 : [0.32787516713142395, 0.8625140190124512]
# acc : 0.863

# MinMaxScaler 적용
# 로스 : [0.32665809988975525, 0.8624837398529053]
# acc : 0.862

# StandardScaler
# 로스 : [0.5132715106010437, 0.7905293107032776]
# acc : 0.791

# MaxAbsScaler
# 로스 : [0.32755860686302185, 0.8618171811103821]
# acc : 0.862

# RobustScaler
# 로스 : [0.3264558017253876, 0.863998532295227]
# acc : 0.864