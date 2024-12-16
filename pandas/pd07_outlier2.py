import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 생성
aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50], 
                [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]]).T
aaa = pd.DataFrame(aaa, columns=["Column1", "Column2"])

# 이상치 탐지 함수 정의
def detect_outliers(data):
    for col in data.columns:
        print(f"=== {col} ===")
        col_data = data[col]
        quartile_1, q2, quartile_3 = np.percentile(col_data, [25, 50, 75])
        
        print("1사분위:", quartile_1)
        print("중앙값(q2):", q2)
        print("3사분위:", quartile_3)
        
        iqr = quartile_3 - quartile_1
        print("IQR:", iqr)
        
        low_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        
        print("하한선:", low_bound)
        print("상한선:", upper_bound)
        
        # 이상치 인덱스 출력
        outliers = col_data[(col_data < low_bound) | (col_data > upper_bound)]
        print("이상치:")
        print(outliers)
        
        # 박스플롯 시각화
        plt.figure(figsize=(6, 4))
        plt.boxplot(col_data, vert=False)
        plt.axvline(low_bound, color='red', linestyle='--', label="Lower Bound")
        plt.axvline(upper_bound, color='blue', linestyle='--', label="Upper Bound")
        plt.title(f"Boxplot of {col}")
        plt.legend()
        plt.show()

# 이상치 탐지 실행
detect_outliers(aaa)

'''
1사분위 : 5.25
q2 :  11.5
3사분위 :  387.5
iqr :  382.25
이상치의 위치 :  (array([ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,
        8,  9,  9, 10, 10, 11, 11, 12, 12], dtype=int64), array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
       0, 1, 0, 1], dtype=int64))

for문


'''



'''








만들기









'''
