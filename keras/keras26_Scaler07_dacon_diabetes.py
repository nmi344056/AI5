# https://dacon.io/competitions/official/236068/mysubmission?isSample=1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

#1. 데이터
path = "C:\\ai5\\_data\\dacon\\diabetes\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape)      # (652, 9)
print(test_csv.shape)       # (116, 8)
print(mission_csv.shape)    # (116, 1)

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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=3434)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train)
print(np.min(x_train), np.max(x_train))     # 0.0 1.0
print(np.min(x_test), np.max(x_test))       # -0.0004456327985739722 0.9798994974874372

#2. 모델구성
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(8,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.2)

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

y_submit = model.predict(test_csv)
print(y_submit.shape)       # (116, 1)

y_submit = np.round(y_submit)
mission_csv['Outcome'] = y_submit
mission_csv.to_csv(path + "sample_submission_0725_19.csv")

print("===============")
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

'''
32 16 16 16 16 1 / train_size=0.8, random_state=3434 / epochs=100, batch_size=16
                loss :  0.1539676934480667 / accuracy :  0.748
MinMaxScaler > loss :  0.1426393687725067 / accuracy :  0.748 > 2
StandardScaler > loss :  0.22998538613319397 / accuracy :  0.718
MaxAbsScaler > loss :  0.12030860036611557 / accuracy :  0.794 > best
RobustScaler > loss :  0.22415265440940857 / accuracy :  0.74
'''
