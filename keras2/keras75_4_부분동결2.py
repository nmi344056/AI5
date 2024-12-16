import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet',
              include_top=True,
              )
'''
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 224, 224, 3)]     0
 block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792
 block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928
 block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0
 block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856
 block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584
 block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0
 block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168
 block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080
 block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080
 block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0
 block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160
 block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808
 block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808
 block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0
 block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808
 block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808
 block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808
 block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0
 flatten (Flatten)           (None, 25088)             0
 fc1 (Dense)                 (None, 4096)              102764544
 fc2 (Dense)                 (None, 4096)              16781312
 predictions (Dense)         (None, 1000)              4097000
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0

                                                            Layer Type    Layer Name  Layer Trainable
0   <keras.engine.input_layer.InputLayer object at 0x000001417F374A60>  input_1       True
1   <keras.layers.convolutional.Conv2D object at 0x00000141536E6790>    block1_conv1  True
2   <keras.layers.convolutional.Conv2D object at 0x00000141536E6D90>    block1_conv2  True
3   <keras.layers.pooling.MaxPooling2D object at 0x0000014153803100>    block1_pool   True
4   <keras.layers.convolutional.Conv2D object at 0x00000141537792E0>    block2_conv1  True
5   <keras.layers.convolutional.Conv2D object at 0x0000014153811760>    block2_conv2  True
6   <keras.layers.pooling.MaxPooling2D object at 0x0000014153779340>    block2_pool   True
7   <keras.layers.convolutional.Conv2D object at 0x0000014153811550>    block3_conv1  True
8   <keras.layers.convolutional.Conv2D object at 0x000001415381E310>    block3_conv2  True
9   <keras.layers.convolutional.Conv2D object at 0x000001415381AE20>    block3_conv3  True
10  <keras.layers.pooling.MaxPooling2D object at 0x00000141538263D0>    block3_pool   True
11  <keras.layers.convolutional.Conv2D object at 0x000001415382C130>    block4_conv1  True
12  <keras.layers.convolutional.Conv2D object at 0x000001415382CF40>    block4_conv2  True
13  <keras.layers.convolutional.Conv2D object at 0x0000014153823160>    block4_conv3  True
14  <keras.layers.pooling.MaxPooling2D object at 0x0000014153836C40>    block4_pool   True
15  <keras.layers.convolutional.Conv2D object at 0x00000141538366D0>    block5_conv1  True
16  <keras.layers.convolutional.Conv2D object at 0x000001415383C040>    block5_conv2  True
17  <keras.layers.convolutional.Conv2D object at 0x000001415383F100>    block5_conv3  True
18  <keras.layers.pooling.MaxPooling2D object at 0x000001415383C730>    block5_pool   True
19  <keras.layers.core.flatten.Flatten object at 0x000001415383CD60>    flatten       True
20  <keras.layers.core.dense.Dense object at 0x000001415387B940>        fc1           True
21  <keras.layers.core.dense.Dense object at 0x000001415387BFA0>        fc2           True
22  <keras.layers.core.dense.Dense object at 0x0000014153835310>        predictions   True
'''

model.layers[17].trainable = False
'''
위는 동일
Total params: 138,357,544
Trainable params: 135,997,736
Non-trainable params: 2,359,808

                                                            Layer Type    Layer Name  Layer Trainable
16  <keras.layers.convolutional.Conv2D object at 0x0000021A1A20B130>    block5_conv2  True
17  <keras.layers.convolutional.Conv2D object at 0x0000021A1A20F0D0>    block5_conv3  False
18  <keras.layers.pooling.MaxPooling2D object at 0x0000021A1A20FE80>    block5_pool   True
19  <keras.layers.core.flatten.Flatten object at 0x0000021A1A20BE80>    flatten       True
20  <keras.layers.core.dense.Dense object at 0x0000021A1A32BA30>        fc1           True
21  <keras.layers.core.dense.Dense object at 0x0000021A1A32BD00>        fc2           True
22  <keras.layers.core.dense.Dense object at 0x0000021A1A3256D0>        predictions   True
'''

model.summary()

import pandas as pd
pd.set_option('max_colwidth', -1)       # Error가 나면 -1을 None으로
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns= ['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
