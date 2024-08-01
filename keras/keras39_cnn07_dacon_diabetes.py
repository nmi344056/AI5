# https://dacon.io/competitions/official/236068/mysubmission?isSample=1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import pandas as pd
import time

#1. 데이터
path = "C:\\ai5\\_data\\dacon\\diabetes\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.columns)
# Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

print(train_csv.info())     # 결측치가 없다
print(test_csv.info())      # 결측치가 없다
print(train_csv.isnull().sum())
print(test_csv.isnull().sum())

x = train_csv.drop(['Outcome'], axis=1)
print(x)                    # [652 rows x 8 columns]
y = train_csv['Outcome']
print(y.shape)              # (652,)

print(np.unique(y, return_counts=True))     
# (array([0, 1], dtype=int64), array([424, 228], dtype=int64))
print(pd.DataFrame(y).value_counts())
# 0          424
# 1          228

x = x.to_numpy()
print(x)
print(x.shape)                      # (652, 8)

x = x/255.
x = x.reshape(652, 8, 1, 1)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=3434)

scaler = MaxAbsScaler()

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(8, 1, 1), strides=1, padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', strides=1, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation='relu', strides=1, padding='same'))
model.add(Flatten())

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=16, input_shape=(32,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))

date = date.strftime("%m%d.%H%M")
print(date)
print(type(date))

path = './_save/keras39/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k39_07_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1, batch_size=16, validation_split=0.2, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
print(y_predict[:20])       # y' 결과
y_predict = np.round(y_predict)
print(y_predict[:20])       # y' 반올림 결과

accuracy_score = accuracy_score(y_test, y_predict)
print("acc score : ", accuracy_score)
print("time : ", round(end - start, 2), "초")

print(test_csv)
test_csv = test_csv.to_numpy()
print(test_csv.shape)       # (116, 8)
test_csv = test_csv.reshape(116, 8, 1, 1)
print(test_csv.shape)

y_submit = model.predict(test_csv)
print(y_submit.shape)       # (116, 1)

y_submit = np.round(y_submit)
mission_csv['Outcome'] = y_submit
mission_csv.to_csv(path + "sample_submission_0801_13.csv")

print("===============")
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

'''
32 16 16 16 16 1 / train_size=0.8, random_state=3434 / epochs=100, batch_size=16


loss :  0.13780251145362854
accuracy :  0.786
'''
