import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
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
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

# model.summary()
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
Trainable params: 14,777,098
Non-trainable params: 0
'''

# vgg16.trainable = False 추가
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
'''

########## [실습] 3가지 비교하기 ##########
# 1. 이전에 본인이 한 최상의 결과
# 2. 가중치를 동결하지 않고 훈련시켰을 때, trainable=True 
# 3. 가중치를 동결하고 훈련시켰을 때, trainable=False
# 위의 2, 3번은 time 체크 까지

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
vgg16.trainable = False   # 가중치 동결
loss :  1.202612280845642
accuracy :  0.584
accuracy_score : 0.1006
time : 97.39 초

vgg16.trainable = True
loss :  2.303286075592041
accuracy :  0.1
accuracy_score : 1.0
time : 126.46 초

'''
