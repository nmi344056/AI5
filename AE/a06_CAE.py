# a03_ae2_그림.py copy
'''
[실습] 모델 구성

padding='same'

encoder                     28
    conv                    28
    maxpool                 14
    conv                    14
    maxpool                 7

decoder
    conv                    7
    UpSampling2D(2, 2)      14
    conv                    14
    UpSampling2D(2, 2)      28
    conv(1, (3,3))          (28, 28, 1)로 만들기

'''

import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype("float32")/255.
x_test = x_test.astype("float32")/255.

                                        # 평균 0,표준편차가 0.1인 정규분포 형태의 랜덤값 

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

print(x_train_noised.shape, x_test_noised.shape)             # (60000, 784) (10000, 784)
print(np.max(x_train), np.min(x_test))                       # 1.0 0.0
print(np.max(x_train_noised), np.min(x_test_noised))         # 1.506013411202829 -0.5281790150375157

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, 0, 1)

#2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, (2,2), input_shape=(28, 28, 1), padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(hidden_layer_size, (2,2), padding='same'))
    model.add(MaxPooling2D())

    model.add(Conv2D(hidden_layer_size, (2,2), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(hidden_layer_size, (2,2), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(1, (3,3), padding='same'))

    return model

# hidden_size = 713  # PCA 1.0일 때
# hidden_size = 486  # PCA 0.999일 때
# hidden_size = 331  # PCA 0.99일 때
hidden_size = 154  # PCA 0.95일 때

autoencoder = autoencoder(hidden_layer_size=hidden_size)

'''
autoencoder.summary()
 Layer (type)                       Output Shape              Param #
======================================================================
 conv2d (Conv2D)                    (None, 28, 28, 154)       770
 max_pooling2d (MaxPooling2D)       (None, 14, 14, 154)      0
 conv2d_1 (Conv2D)                  (None, 14, 14, 154)       95018
 max_pooling2d_1 (MaxPooling 2D)    (None, 7, 7, 154)        0
 conv2d_2 (Conv2D)                  (None, 7, 7, 154)         95018
 up_sampling2d (UpSampling2D)       (None, 14, 14, 154)      0
 conv2d_3 (Conv2D)                  (None, 14, 14, 154)       95018
 up_sampling2d_1 (UpSampling 2D)    (None, 28, 28, 154)      0
 conv2d_4 (Conv2D)                  (None, 28, 28, 1)         1387
=================================================================
Total params: 287,211
Trainable params: 287,211
Non-trainable params: 0
'''

#3. 컴파일, 훈련
autoencoder.compile(optimizer='adam', loss ='mse')
# autoencoder.compile(optimizer='adam', loss ='binary_crossentropy')

autoencoder.fit(x_train_noised, x_train, epochs=30, batch_size=128, validation_split=0.2)

#4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test_noised)

import matplotlib.pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = \
      plt.subplots(3, 5, figsize=(20,7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(decoded_imgs.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈를 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('NOISE', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 맨 아래에 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_imgs[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
