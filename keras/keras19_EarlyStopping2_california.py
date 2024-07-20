from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)         # (20640, 8) (20640,)

#[실습] 만들기 R2 성능 0.59 이상
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123)

#2. 모델구성
model = Sequential()
model.add(Dense(60, input_dim=8))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_besr_weights=True)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.2, verbose=3, callbacks=[es])
end = time.time()

#4. 평가,예측
print("++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)
print("time : ", round(end - start, 2), "초")

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('캘리포니아 Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()

'''
3 4 5 3 1
train_size=0.7, random_state=123 / 
loss :  0.588417649269104
r2 score :  0.5550008546448328
'''

print("++++++++++++++++++++")
