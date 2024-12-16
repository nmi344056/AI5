import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100

vgg16 = VGG16(# weights='imagenet',
              include_top=False,
              input_shape=(100, 100, 3),
              )

# vgg16.trainable = False     # 가중치 동결 
vgg16.trainable = True

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))

model.summary()

### 실습 ###
# 비교할거 
# 1. 이전의 본인이 한 최상의 겨로가
# 2. 가중치를 동결하지 않고 훈련시켰을때, trainable=True 
# 3. 가중치를 동결하고 훈련시켰을 때, trainable=False
# 시간까지 비교 하기 

np_path = "C:\\ai5\\_data\\_save_npy\\keras45\\"
x_train = np.load(np_path + 'keras45_02_horse_x_train.npy')
y_train = np.load(np_path + 'keras45_02_horse_y_train.npy')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=921)

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

##### 스케일링
# x_train = x_train/255.      # 0~1 사이 값으로 바뀜
# x_test = x_test/255.

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=10,
          verbose=1, 
          validation_split=0.2,
          callbacks=[es],
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],2))

y_pre = model.predict(x_test)
print("걸린 시간 :", round(end-start,2),'초')

'''
vgg16.trainable = False   # 가중치 동결
loss : 0.0036818261723965406
acc : 1.0
걸린 시간 : 14.03 초

vgg16.trainable = True
loss : 0.017385829240083694
acc : 0.99
걸린 시간 : 44.6 초

'''
