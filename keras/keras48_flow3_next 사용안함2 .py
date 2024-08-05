import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,     # 처음부터 0~1 사이의 스케일링한 데이터를 달라
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.2,  # 평행 이동, 10%만큼 옆으로 이동
    # height_shift_range=0.1, # 평행 이동 수직
    rotation_range=15,       # 정해진 각도만큼 이미지 회전
    # zoom_range=1.2,         # 축소 또는 확대
    # shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (일그러짐)
    fill_mode='nearest',    # 주변의 이미지의 비율대로 채운다
)

augment_size = 100

print(x_train.shape)        # (60000, 28, 28)
print(x_train[0].shape)     # (28, 28)

# plt.imshow(x_train[0], cmap='gray')
# plt.show()

# aaa = np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1)
# print(aaa.shape)            # (100, 28, 28, 1)

# aaa = np.tile(x_train[0], augment_size).reshape(-1, 28, 28, 1)
# print(aaa.shape)            # (100, 28, 28, 1) (버전이 올라가면서 reshape 안해도 된다?)

xy_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),
    np.zeros(augment_size),
    batch_size=32,    # 주석하면 print(xy_data[0][0].shape)가 (32, 28, 28, 1), default=32
    shuffle=False,
)   # .next()

# [검색] np.tile

print(xy_data)              # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001952FF4A190>
print(type(xy_data))        # <class 'keras.preprocessing.image.NumpyArrayIterator'>, .next()가 없으면 
# print(xy_data.shape)      # AttributeError: 'NumpyArrayIterator' object has no attribute 'shape'
print(len(xy_data))         # 4

# print(xy_data[0].shape)   # AttributeError: 'tuple' object has no attribute 'shape'
print(xy_data[0][0].shape)  # (32, 28, 28, 1)
print(xy_data[3][0].shape)  # (4, 28, 28, 1)
# print(xy_data[4][0].shape)  # ValueError: Asked to retrieve element 4, but the Sequence has length 4

print(xy_data[0][1].shape)  # (32,) y, .next()있을 때 print(xy_data[1].shape)

# plt.figure(figsize=(7,7))   #사이즈는 봐가면서 조절
# for i in range(49):
#     plt.subplot(7, 7, i+1)  #(7,7)의 1번째, ..., 순차 증가
#     plt.imshow(xy_data[0][0][i], cmap='gray')
    
# plt.show()
