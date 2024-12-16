import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import time

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1) 컬러 데이터
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

##### 스케일링 #####
x_train = x_train/255.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) # 1.0 0.0

#2. 모델 구성
vgg16 = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(32, 32 ,3),
              )

vgg16.trainable = False     # 가중치 동결
# vgg16.trainable = True

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()
'''
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688
 flatten (Flatten)           (None, 512)               0
 dense (Dense)               (None, 100)               51300
 dense_1 (Dense)             (None, 100)               10100
 dense_2 (Dense)             (None, 10)                1010
=================================================================
Total params: 14,777,098
Trainable params: 62,410
Non-trainable params: 14,714,688

 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688     # (1,1) 이여서 차이가 없다
 global_average_pooling2d (G  (None, 512)              0
 lobalAveragePooling2D)
 dense (Dense)               (None, 100)               51300
 dense_1 (Dense)             (None, 100)               10100
 dense_2 (Dense)             (None, 10)                1010
=================================================================
Total params: 14,777,098
Trainable params: 62,410
Non-trainable params: 14,714,688
'''

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

start = time.time()

hist =model.fit(x_train, y_train,
                validation_split=0.2,
                epochs=1000,
                batch_size=100, 
                verbose=1,
                callbacks=[es],
                )

end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("accuracy : ", round(loss[1], 3))

y_predict = model.predict(x_test)
# print(y_predict)            # float 형
# print(y_predict.shape)      # (10000, 10)

y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

acc = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc)
print("time :", round(end-start,2),'초')

'''
성능 비교
'''
