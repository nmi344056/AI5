import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
# text = '블라블라 블라 블라블라 라라라라라라라'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9}

print(token.word_counts)
# OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 4), ('먹었다', 1)])

x = token.texts_to_sequences([text])    #리스트
print(x)            # [[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]
# print(x.shape)      # 리스트는 shape 없다, (1,9)

# [실습] OneHot Encoding 3가지 만들기
# 1부터 시작 주의

from tensorflow.keras.utils import to_categorical
'''
# x_ohe1 = to_categorical(x)
# x_ohe1 = x_ohe1[:, :, 1:].reshape(14, 9)
'''
x_ohe1 = to_categorical(x, num_classes=10)
print(x_ohe1, x_ohe1.shape)
'''
[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1.]] (14, 9)
'''

'''
x = [item for sublist in x for item in sublist]
x = pd.Series(x)
'''
# x = np.array(x).reshape(-1, )
# x_ohe2 = pd.get_dummies(x)          # pandas
# print(x_ohe2, x_ohe2.shape)
'''
    1  2  3  4  5  6  7  8  9
0   0  0  0  1  0  0  0  0  0
1   0  0  0  0  1  0  0  0  0
2   0  1  0  0  0  0  0  0  0
3   0  1  0  0  0  0  0  0  0
4   0  0  1  0  0  0  0  0  0
5   0  0  1  0  0  0  0  0  0
6   0  0  0  0  0  1  0  0  0
7   0  0  0  0  0  0  1  0  0
8   0  0  0  0  0  0  0  1  0
9   1  0  0  0  0  0  0  0  0
10  1  0  0  0  0  0  0  0  0
11  1  0  0  0  0  0  0  0  0
12  1  0  0  0  0  0  0  0  0
13  0  0  0  0  0  0  0  0  1 (14, 9)
'''

# print("==============================")
# from sklearn.preprocessing import OneHotEncoder
# x = np.array(x).reshape(-1, 1)      # (150, 1)
# ohe = OneHotEncoder(sparse=False)   # True가 default
# x_ohe3 = ohe.fit_transform(x)
# print(x_ohe3, x_ohe3.shape)
'''
[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1.]] (14, 9)
'''





