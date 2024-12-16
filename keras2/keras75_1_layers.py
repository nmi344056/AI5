import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet',
              include_top=False,
              input_shape=(32, 32, 3),
              )

vgg16.trainable = False     # VGG=False (30, 4)

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

# vgg16.trainable = False     # model=False (30, 0)
model.summary()
'''
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688
 flatten (Flatten)           (None, 512)               0
 dense (Dense)               (None, 100)               51300
 dense_1 (Dense)             (None, 10)                1010
=================================================================
Total params: 14,766,998
Trainable params: 0
Non-trainable params: 14,766,998
'''

print(len(model.weights))
print(len(model.trainable_weights))

'''
                            trainable=True   model=False   VGG=False
len(model.weights)              30              30              30
len(model.trainable_weights)    30              0               4

'''
