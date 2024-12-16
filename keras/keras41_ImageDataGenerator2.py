'''
배치를 16으로 잡고
x, y를 추출해서 모델 만들기
acc 0.99 이상

batch_size=160
x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]
'''

import numpy as np
import pandas as pd
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

train_datagen = ImageDataGenerator(rescale=1./255,)
test_datagen = ImageDataGenerator(rescale=1./255,)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(200,200),  # resize, 동일한 규격 사용
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)   # Found 160 images belonging to 2 classes.  (ad, normal 합 총 160개)

xy_test = test_datagen.flow_from_directory(
    path_train,
    target_size=(200,200),  # resize, 동일한 규격 사용
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale',
)   # Found 160 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape)         # (160, 200, 200, 1) (160,)
print(x_test.shape, y_test.shape)           # (160, 200, 200, 1) (160,)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(8, (2,2), activation='relu'))
model.add(Flatten())

model.add(Dense(units=8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=4, input_shape=(32,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))

date = date.strftime("%m%d.%H%M")
print(date)
print(type(date))

path = './_save/keras41/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k41_02_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=160, validation_split=0.2, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
# print(y_predict)            # float 형
# print(y_predict.shape)      # (10000, 10)

# acc = accuracy_score(y_test, y_predict)
# print('accuracy_score :', acc)
# print("time :", round(end-start,2),'초')

'''
[실습] accuracy 0.98 이상
loss :  0.02529209852218628 / accuracy :  1.0 > k41_02_date_0805.1230_epo_0049_valloss_0.0479

'''
