import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=555)
print(x, x.shape)       # (506, 13)
print(y, y.shape)       # (506,)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(30))
model.add(Dense(500))
model.add(Dense(30))
model.add(Dense(100))
model.add(Dense(30))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=30, validation_split=0.2, verbose=3)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)
print("time : ", round(end - start, 2), "초")

# print("++++++++++ hist ++++++++++")
# print(hist)     # <keras.callbacks.History object at 0x0000024F10C4E550>

# print("++++++++++ hist.history ++++++++++")
# print(hist.history)
# # {'loss': [2719.207763671875, 237.2079315185547, 135.118896484375, 105.3738021850586, 80.42974853515625, 81.15362548828125, 80.77424621582031, 75.98873138427734, 76.49324798583984, 76.45459747314453], 'val_loss': [374.59197998046875, 194.48721313476562, 106.5069808959961, 86.67974853515625, 81.953857421875, 
# # 82.79047393798828, 79.17298889160156, 79.05679321289062, 76.53048706054688, 78.53487396240234]}

# print("++++++++++ loss ++++++++++")
# print(hist.history['loss'])
# print("++++++++++ val_loss ++++++++++")
# print(hist.history['val_loss'])

print("++++++++++++++++++++")
import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('보스턴 Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()

'''
train_size=0.7, random_state=555 / epochs=500, batch_size=10, validation_split=0.2
loss :  17.924062728881836
r2 score :  0.7725105035697983
'''
