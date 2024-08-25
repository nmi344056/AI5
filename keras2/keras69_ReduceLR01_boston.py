# 68_1 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston

# seed를 고정하고 성능 비교
import tensorflow as tf
tf.random.set_seed(337)     # tensorflow seed 고정

np.random.seed(337)         # numpy seed 고정

import random as rn
rn.seed(337)                # python seed 고정

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=123)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1, restore_best_weights=True,)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=15, verbose=1, factor=0.9)

from tensorflow.keras.optimizers import Adam
learning_rate = 0.00001        # Default : 0.001
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es, rlr])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=0)
# print('lr : {0}, loss : {1}'.format(learning_rate, loss))

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
# print('lr : {0}, r2 : {1}'.format(learning_rate, r2))

print('{0} > loss : {1} / r2 : {2}'.format(learning_rate, loss, r2))

'''
learning_rate
        > loss :25.424827575683594 / r2 :0.6779312699071509
1       > loss : 8123.08544921875 / r2 : -101.89910075708643
0.1     > loss : 26.737720489501953 / r2 : 0.6613002156818426
0.01    > loss :25.356815338134766 / r2 :0.6787928447535494
0.005   > loss :25.61310577392578 / r2 :0.6755462435925752
0.001   > loss :25.424823760986328 / r2 :0.6779312841214709     Default
0.0007  > loss :25.3327693939209 / r2 :0.6790974089503052       ***
0.0001  > loss :334.2796936035156 / r2 :-3.234484516344792

ReduceLROnPlateau / factor=0.9
0.1     > loss : 25.747495651245117 / r2 : 0.6738438644259267
0.01    > loss : 25.21685791015625 / r2 : 0.6805657219676258
0.005   > loss : 25.83310317993164 / r2 : 0.6727594141775739
0.001   > loss : 25.176774978637695 / r2 : 0.6810734547230621
0.0007  > loss : 25.26230239868164 / r2 : 0.6799900446826969
0.0001  > loss : 24.93173599243164 / r2 : 0.6841774903112595     ***

'''
