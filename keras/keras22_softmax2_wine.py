import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
datasets = load_wine()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (178, 13) (178,)

print(y)
print(np.unique(y, return_counts=True))
print(pd.value_counts(y))

y = pd.get_dummies(y)
print(y.shape)      # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9,
                                                    random_state=2321, stratify=y)

#2. 모델구성
model = Sequential()
model.add(Dense(8, activation='relu', input_dim=13))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))

model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 100,
    restore_best_weights = True
)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=4,
                 validation_split=0.2, callbacks=[es])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)
print(y_pred)
y_pred = np.round(y_pred)
print(y_pred)

accuracy_score = accuracy_score(y_test, y_pred)
print('acc_score :', accuracy_score)
print('걸린시간 :', round(end - start, 2), '초')
print('로스 :', loss)
print('acc :', round(loss[1],3))

# 걸린시간 : 25.55 초
# 로스 : [0.0890042781829834, 1.0]
# acc : 1.0























