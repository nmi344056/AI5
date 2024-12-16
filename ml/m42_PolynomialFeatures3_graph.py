import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
plt.rcParams['font.family'] = 'Malgun Gothic'

#1. 데이터
np.random.seed(777)
x = 2 * np.random.rand(100, 1) -1   # -1부터 1까지 난수 100개 생성
print(x)
print(np.max(x), np.min(x))         # 0.9991190865361039 -0.9852230982722201

y = 3 * x**2 + 2 * x + 1 + np.random.randn(100, 1) # y = 3x^2 + 2x + 1 + 노이즈

# rand : 0~1 사이의 균등 분포 난수
# randn : 평균 0, 표준편차 1의 정규 분포 난수

pf = PolynomialFeatures(degree=2, include_bias=False)
x_poly = pf.fit_transform(x)
print(x_poly)   # (x^2)

#2. 모델
model = LinearRegression()
model2 = LinearRegression()

#3. 훈련
model.fit(x, y)
model2.fit(x_poly, y)

# 원래 데이터 그래프
plt.scatter(x, y, color='blue', label='Original Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression 예제')
# plt.show()

# 다항식 회귀 그래프
x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
x_test_poly = pf.transform(x_test)
y_plot = model.predict(x_test)
y_plot2 = model2.predict(x_test_poly)
plt.plot(x_test, y_plot, color='red', label='Linear Regression')
plt.plot(x_test, y_plot2, color='green', label='Polynomial Regression')

plt.legend()
plt.show()
