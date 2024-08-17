import numpy as np
import pandas as pd
from tensorflow.keras.layers import Concatenate, concatenate
from tensorflow.keras.preprocessing.text import Tokenizer

text1 = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
text2 = '태운이는 선생을 괴롭힌다. 준영이는 못생겼다. 사영이는 마구 마구 더 못생겼다.'

# [실습] 리스트형태로 OneHot Encoding 3가지 만들기
token = Tokenizer()
token.fit_on_texts([text1, text2])

# print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '못생겼다': 4, '나는': 5, '지금': 6, '맛있는': 7, '김밥을': 8, '엄청': 9, '먹었다': 10, '태운이는': 11, '선생을': 12, '괴롭힌다': 13, '준영이는': 14, '사영이는': 15, '더': 16}

print(token.word_counts)
# OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 6), ('먹었다', 1), ('태운이는', 1), ('선생을', 1), ('괴롭힌다', 1), ('준영이
# 는', 1), ('못생겼다', 2), ('사영이는', 1), ('더', 1)])

x = token.texts_to_sequences([text1, text2])    #리스트
print(x)            # [[5, 6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 1, 10], [11, 12, 13, 14, 4, 15, 1, 1, 16, 4]]
# print(x.shape)      # 리스트는 shape 없다, 

text = np.concatenate(x)

from tensorflow.keras.utils import to_categorical
x_ohe1 = to_categorical(x)
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
x = np.array(x).reshape(-1, )
x_ohe2 = pd.get_dummies(x)          # pandas
print(x_ohe2, x_ohe2.shape)
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





