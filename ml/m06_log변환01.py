import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale=2.0, size=1000)      #1000개의 데이터 생성
print(data)
print(data.shape)                       # (1000,)
print(np.min(data), np.max(data))       # 0.0009614110702811401 9.1374106990532

# log_data = np.log(data)
log_data = np.log1p(data)               # log0은 무한대, 이를 방지하기 위해 +1 (log1p)

# 원본 데이터 히스토그램
plt.subplot(1, 2, 1)                    # 1과 2짜리의 첫번째 그림
plt.hist(data, bins=50, color='blue', alpha=0.5)
plt.title('Original')
# plt.show()

# 로그변환 데이터 히스토그램
plt.subplot(1, 2, 2)                    # 1과 2짜리의 두번째 그림
plt.hist(log_data, bins=50, color='red', alpha=0.5)
plt.title('Log Transformed')
plt.show()

# 지수변환 원래대로 되돌리기
exp_data = np.exp(log_data)-1           # log1p는 exp()-1
