# 46 copy

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img       # 이미지 가져오기
from tensorflow.keras.preprocessing.image import img_to_array   # 가져온 이미지 수치화

path = 'C:\\ai5\\_data\\image\\me\\a.jpg'       #.png도 가능

img = load_img(path, target_size=(100, 100),)
print(img)          # <PIL.Image.Image image mode=RGB size=200x200 at 0x2D3B261A6A0>
print(type(img))    # <class 'PIL.Image.Image'>

# plt.imshow(img)
# plt.show()          # 사진 출력

arr = img_to_array(img)
print(arr)
print(arr.shape)    # (200, 200, 3) -> (100, 100, 3)
print(type(arr))    # <class 'numpy.ndarray'>

#차원 증가
img = np.expand_dims(arr, axis=0)   #이렇게도 가능
print(img.shape)    # (1, 100, 100, 3)

########## 증폭 ##########

datagen = ImageDataGenerator(
    rescale=1./255,     # 처음부터 0~1 사이의 스케일링한 데이터를 달라
    # horizontal_flip=True,   # 수평 뒤집기
    # vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.2,  # 평행 이동, 10%만큼 옆으로 이동
    # height_shift_range=0.1, # 평행 이동 수직
    rotation_range=15,       # 정해진 각도만큼 이미지 회전
    # zoom_range=1.2,         # 축소 또는 확대
    # shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (일그러짐)
    fill_mode='nearest',    # 주변의 이미지의 비율대로 채운다
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

it = datagen.flow(img,       #수치화 된 데이터
             batch_size=1,
             
             )
# print(it)               # <keras.preprocessing.image.NumpyArrayIterator object at 0x0000021B8AE89460>
# print(it.next())

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 10))

for i in range(5):
    batch = it.next()
    print(batch.shape)  # (1, 100, 100, 3) -> reshape 한다
    batch = batch.reshape(100, 100, 3)
    
    ax[i].imshow(batch)
    ax[i].axis('off')
    
plt.show()
