# 74_2 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(# weights='imagenet', 
              include_top=False,
              input_shape=(224, 224 ,3),
              )

vgg16.trainable = False   # 가중치 동결

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
 vgg16 (Functional)          (None, 7, 7, 512)         14714688

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

# [실습] Flatten과 GAP 비교
