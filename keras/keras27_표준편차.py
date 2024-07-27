from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import numpy as np

#1. data
data = np.array([[1,2,3,1],
                 [4,5,6,2],
                 [7,8,9,3],
                 [10,11,12,114],
                 [13,14,15,115]])

#1-1. 평균
means = np.mean(data, axis=0)
print('col 평균 : ', means)                    # 평균 :  [ 7.  8.  9. 47.]

#1-2. 모집단 분산 (n빵)
population_variances = np.var(data, axis=0)    # ddof=0 default
print('모집단 분산 : ', population_variances)   #  모집단 분산 :  [  18.   18.   18. 3038.]

#1-3. 포본 분산 (n-1 빵)
variances = np.var(data, axis=0, ddof=1)
print('포본 분산 : ', variances)                # 포본 분산 :  [  22.5   22.5   22.5 3797.5]

#1-4. 포본 표준편차 (루트-표본 분산)
std = np.std(data, axis=0, ddof=1)
print('포본 표준편차 : ', std)                  # 포본 표준편차 :  [ 4.74341649  4.74341649  4.74341649 61.62385902]

#1-5. StandardScaler                           (x-평균)/표준편차
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print('StandarsScaler : \n', scaled_data)
# StandarsScaler :
#  [[-1.41421356 -1.41421356 -1.41421356 -0.83457226]
#  [-0.70710678 -0.70710678 -0.70710678 -0.81642939]
#  [ 0.          0.          0.         -0.79828651]
#  [ 0.70710678  0.70710678  0.70710678  1.21557264]
#  [ 1.41421356  1.41421356  1.41421356  1.23371552]]
