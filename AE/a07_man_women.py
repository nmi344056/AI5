'''
[실습] 만들기

keras45_07_save_npy_gender.py
keras45_08_load_npy_gender
keras45_09_save_npy_image_cat_dog
keras46_img_to_array
keras47_01_내가개가될상인가
keras47_02_내가남자게여자게

참고해서 사진에 노이즈를 주고 
오토인코더로 피부미백 훈련 가중치를 만든다.
그 가중치로 내 사진을 predict 해서 피부미백 시킨다.

출력 이미지는 (원본, 노이즈, predict) 순으로 출력

'''

import numpy as np
import tensorflow as tf
import PIL.Image as Image
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
path_np = 'C:\\ai5\\_data\\_save_npy\\keras45\\'
x_train = np.load(path_np + 'keras45_07_gender_x_train.npy')

# me_path_np = 'C:\\ai5\\_data\\image\\me\\'
# x_test = np.load(me_path_np + 'keras46_01_me_x_train.npy')

# x_train = x_train.astype("float32")/255.      # 이거 하면 훈련이 제대로 안된다. output이 초록색으로 나옴.
# x_test = x_test.astype("float32")/255.

x_test = np.array(Image.open('./_data/image/me/a.jpg').resize((100, 100))).reshape(1, 100, 100, 3) / 255.

x_train_noised = x_train[:5000] + np.random.normal(0, 0.1, size=x_train[:5000].shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

print(x_train_noised.shape, x_test_noised.shape)             # (27167, 100, 100, 3) (1, 100, 100, 3)
print(np.max(x_train), np.min(x_test))                       # 1.0 0.0
print(np.max(x_train_noised), np.min(x_test_noised))         # 0.5946239770942013 -0.30889855251302895

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, 0, 1)

#2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, (2,2), input_shape=(100, 100, 3), padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(hidden_layer_size, (2,2), padding='same'))
    model.add(MaxPooling2D())

    model.add(Conv2D(hidden_layer_size, (2,2), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(hidden_layer_size, (2,2), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(3, (3,3), padding='same'))

    return model

# hidden_size = 713  # PCA 1.0일 때
# hidden_size = 486  # PCA 0.999일 때
# hidden_size = 331  # PCA 0.99일 때
# hidden_size = 154  # PCA 0.95일 때
hidden_size = 128  # PCA 0.95일 때

autoencoder = autoencoder(hidden_layer_size=hidden_size)

#3. 컴파일, 훈련
autoencoder.compile(optimizer='adam', loss ='mse')
# autoencoder.compile(optimizer='adam', loss ='binary_crossentropy')

autoencoder.fit(x_train_noised, x_train, epochs=100, batch_size=16, validation_split=0.2)

#4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test_noised)

import matplotlib.pyplot as plt
import random

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,7))

# 원본(입력) 이미지를 맨 위에 그린다.
ax1.imshow(x_test[0])
ax1.set_ylabel('INPUT', size=20)
ax1.grid(False)
ax1.set_xticks([])
ax1.set_yticks([])

# 노이즈를 넣은 이미지
ax2.imshow(x_test_noised[0])
ax2.set_ylabel('NOISE', size=20)
ax2.grid(False)
ax2.set_xticks([])
ax2.set_yticks([])

# 오토인코더가 출력한 이미지를 맨 아래에 그린다.
ax3.imshow(decoded_imgs[0])
ax3.set_ylabel('OUTPUT', size=20)
ax3.grid(False)
ax3.set_xticks([])
ax3.set_yticks([])

plt.tight_layout()
plt.show()
