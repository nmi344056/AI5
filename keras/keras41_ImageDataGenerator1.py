import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,     # 처음부터 0~1 사이의 스케일링한 데이터를 달라
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.1,  # 평행 이동, 10%만큼 옆으로 이동
    height_shift_range=0.1, # 평행 이동 수직
    rotation_range=5,       # 정해진 각도만큼 이미지 회전
    zoom_range=1.2,         # 축소 또는 확대
    shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (일그러짐)
    fill_mode='nearest',    # 주변의 이미지의 비율대로 채운다
)

test_datagen = ImageDataGenerator(
    rescale=1./255,         # 평가를 제대로 하기위해 
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(200,200),  # resize, 동일한 규격 사용
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)

# [F5] -> Found 160 images belonging to 2 classes.  (ad, normal 합 총 160개)

xy_test = test_datagen.flow_from_directory(
    path_train,
    target_size=(200,200),  # resize, 동일한 규격 사용
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
)

# [F5] -> Found 160 images belonging to 2 classes.  (ad, normal 합 총 160개)

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x00000188B3BA5700>
# print(xy_train.next())  # array([0., 0., 0., 0., 1., 1., 1., 0., 1., 1.], dtype=float32))
# print(xy_train.next())  # array([0., 0., 1., 0., 1., 0., 0., 0., 0., 0.], dtype=float32))

print(xy_train[0])
print(xy_train[0][0])
print(xy_train[0][1])

# print(xy_train[0].shape)        # AttributeError: 'tuple' object has no attribute 'shape'
print(xy_train[0][0].shape)     # (10, 200, 200, 1)
print(xy_train[0][1].shape)     # (10,)

# print(xy_train[16])             # ValueError: Asked to retrieve element 16, but the Sequence has length 16
# print(xy_train[15][2])          # IndexError: tuple index out of range

print(type(xy_train))           # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))        # <class 'tuple'>
print(type(xy_train[0][0]))     # <class 'numpy.ndarray'>
print(type(xy_train[0][1]))     # <class 'numpy.ndarray'>
