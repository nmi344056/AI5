
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.summary()
#  Layer (type)                Output Shape              Param #(파라미터=연산량)
# =================================================================
#  dense (Dense)               (None, 3)                 6

#  dense_1 (Dense)             (None, 4)                 16

#  dense_2 (Dense)             (None, 3)                 15

#  dense_3 (Dense)             (None, 1)                 4
# input 은 표시 안됨
# ==============================================================================
# Total params: 41              
# Trainable params: 41          
# Non-trainable params: 0       사전학습, 훈련을 시키지 않고 훈련값 가져오기?
