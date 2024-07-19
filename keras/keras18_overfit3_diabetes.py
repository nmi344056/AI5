from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)     #(442, 10) (442,)

#[실습] 맹그러봐
# R2 0.62 이상

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9,
                                                    random_state= 8000)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
hist = model.fit(x_train, y_train, validation_split=0.25,
           epochs=1500, batch_size=15)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 스코어: ", r2)
print("걸린시간 :", round(end-start, 2), "초")

print("============================= hist =========================")
print(hist)
print("============================= hist.history ==================")
print(hist.history)
print("============================= loss ==================")
print(hist.history['loss'])
print("============================= val_loss ==================")
print(hist.history['val_loss'])
print("==========================================================")

import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.figure(figsize=(9,6))       # 그림판 사이즈
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('당뇨 loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()

# 로스 :  2291.559326171875
# r2 스코어:  0.6161708204975695
# 로스 :  2285.40478515625
# r2 스코어:  0.6172017509215408