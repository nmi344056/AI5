import pandas as pd
import numpy as np

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan,],
                     [2, 4, 6, 8, 10,],
                     [np.nan, 4, np.nan, 8, np.nan,],
                     ])
# print(data)

data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
# print(data)
'''
     x1   x2    x3   x4
0   2.0  2.0   2.0  NaN
1   NaN  4.0   4.0  4.0
2   6.0  NaN   6.0  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
'''

from sklearn.impute import SimpleImputer, KNNImputer

imputer = SimpleImputer()                    # Default : 평균
data2 = imputer.fit_transform(data)
# print(data2)
'''
[[ 2.          2.          2.          6.        ]
 [ 6.5         4.          4.          4.        ]
 [ 6.          4.66666667  6.          6.        ]
 [ 8.          8.          8.          8.        ]
 [10.          4.66666667 10.          6.        ]]
 '''

imputer = SimpleImputer(strategy='mean')     # 평균
data3 = imputer.fit_transform(data)
# print(data3)
'''
[[ 2.          2.          2.          6.        ]
 [ 6.5         4.          4.          4.        ]
 [ 6.          4.66666667  6.          6.        ]
 [ 8.          8.          8.          8.        ]
 [10.          4.66666667 10.          6.        ]]
'''

imputer = SimpleImputer(strategy='median')   # 중위
data4 = imputer.fit_transform(data)
# print(data4)
'''
[[ 2.  2.  2.  6.] 
 [ 7.  4.  4.  4.] 
 [ 6.  4.  6.  6.] 
 [ 8.  8.  8.  8.] 
 [10.  4. 10.  6.]]
'''

imputer = SimpleImputer(strategy='most_frequent')   # 최빈값, 가장 자주 나온다
data5 = imputer.fit_transform(data)
# print(data5)
'''
[[ 2.  2.  2.  4.]
 [ 2.  4.  4.  4.]
 [ 6.  2.  6.  4.]
 [ 8.  8.  8.  8.]
 [10.  2. 10.  4.]]
'''

imputer = SimpleImputer(strategy='constant', fill_value=777)   # 상수, 특정값
data6 = imputer.fit_transform(data)
# print(data6)
'''
[[  2.   2.   2. 777.]
 [777.   4.   4.   4.]
 [  6. 777.   6. 777.]
 [  8.   8.   8.   8.]
 [ 10. 777.  10. 777.]]
'''

imputer = KNNImputer()   # KNN 알고리즘으로 결측치 처리
data7 = imputer.fit_transform(data)
# print(data7)
'''
[[ 2.          2.          2.          6.        ]
 [ 6.5         4.          4.          4.        ]
 [ 6.          4.66666667  6.          6.        ]
 [ 8.          8.          8.          8.        ]
 [10.          4.66666667 10.          6.        ]]
'''
