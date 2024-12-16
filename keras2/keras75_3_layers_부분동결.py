import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet',
              include_top=False,
              input_shape=(32, 32, 3),
              )

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

#1. 전체 동결
# model.trainable = False
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

                                                          Layer Type Layer Name  Layer Trainable
0  <keras.engine.functional.Functional object at 0x0000018A6893A0A0>  vgg16      False
1  <keras.layers.core.flatten.Flatten object at 0x0000018A6893A370>   flatten    False
2  <keras.layers.core.dense.Dense object at 0x0000018A689405E0>       dense      False
3  <keras.layers.core.dense.Dense object at 0x0000018A689E85B0>       dense_1    False
'''

#2. 전체 동결
# for layer in model.layers:
#     layer.trainable = False
'''
위와 동일
'''

#3. 부분 동결
'''
print(model.layers)
# [<keras.engine.functional.Functional object at 0x000001FFF13D90A0>, <keras.layers.core.flatten.Flatten object at 0x000001FFF13D9370>, <keras.layers.core.dense.Dense object at 0x000001FFF13DF5E0>, <keras.layers.core.dense.Dense object at 0x000001FFF14475B0>]

print(model.layers[0])      # <keras.engine.functional.Functional object at 0x000002B9FC40A340>
print(model.layers[1])      # <keras.layers.core.flatten.Flatten object at 0x0000024F392AA280>
print(model.layers[2])      # <keras.layers.core.dense.Dense object at 0x000002437CA94EE0>

model.layers[1].trainable = False
                                                          Layer Type Layer Name  Layer Trainable
0  <keras.engine.functional.Functional object at 0x000001441CEB90D0>  vgg16      True
1  <keras.layers.core.flatten.Flatten object at 0x000001441CEB94F0>   flatten    False
2  <keras.layers.core.dense.Dense object at 0x000001441CEC0610>       dense      True
3  <keras.layers.core.dense.Dense object at 0x00000144243875E0>       dense_1    True
'''

model.layers[0].trainable = False
'''
                                                          Layer Type Layer Name  Layer Trainable
0  <keras.engine.functional.Functional object at 0x000002265B2B90D0>  vgg16      False
1  <keras.layers.core.flatten.Flatten object at 0x000002265B2B94F0>   flatten    True
2  <keras.layers.core.dense.Dense object at 0x000002265B2C0610>       dense      True
3  <keras.layers.core.dense.Dense object at 0x000002265B6485E0>       dense_1    True
'''

model.summary()

import pandas as pd
pd.set_option('max_colwidth', -1)       # Error가 나면 -1을 None으로
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns= ['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
