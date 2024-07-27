from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston
import numpy as np
import time

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=555)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(32, input_shape=(13,)))
# model.add(Dense(16))
# model.add(Dense(16))
# model.add(Dense(16))
# model.add(Dense(1))

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True,)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

# start = time.time()
# hist = model.fit(x_train, y_train, epochs=500, batch_size=10, validation_split=0.2, callbacks=[es, mcp])
# end = time.time()

#4. 평가, 예측
print('========== 2. mcp 출력 ==========')
model = load_model('./_save/keras30_mcp/k30_01_boston_date_0726.2040_epo_0032_valloss_17.7942.hdf5')
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 score : ", r2)

'''
32 16 16 16 1 / train_size=0.7, random_state=555 / epochs=500, batch_size=10

========== save ==========
Epoch 82/500
 1/33 [..............................] - ETA: 0s - loss: 14.2641Restoring model weights from the end of the best epoch: 32.

Epoch 00082: val_loss did not improve from 17.79425
33/33 [==============================] - 0s 750us/step - loss: 25.9872 - val_loss: 20.2842
Epoch 00082: early stopping
4/4 [==============================] - 0s 333us/step - loss: 18.4880
loss :  18.48796844482422
r2 score :  0.7653534998534318

========== mcp 출력 ==========
loss :  18.48796844482422
r2 score :  0.7653534998534318
'''
