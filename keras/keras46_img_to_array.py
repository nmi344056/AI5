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

#[실습] me 폴더 위에 데이터를 npy로 저장할 것

path_np = 'C:\\ai5\\_data\\image\\me\\'
np.save(path_np + 'keras46_01_me_x_train.npy', arr=img)
