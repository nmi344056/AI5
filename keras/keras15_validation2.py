import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

## 잘라라!!!

x_train = x[ :10]
y_train = y[ :10]

x_val = x[10:13]
y_val = y[10:13]

x_test = x[13:17]
y_test = y[13:17]

# x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9, 
#                                                     random_state=8743)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train ,train_size=0.2,
#                                                   random_state=8743)
print(x_train, y_train, x_val, y_val)
'''
#2. 모델구성

model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1,
          verbose=1,        #verbode=0 바로출력, 1 디폴트, 2 진행바 삭제, 3이상 epochs만 출력
          validation_data=(x_val, y_val),      # 이 파일에서 요놈만 추가됨        
          )
# verbose=0 : 침묵
# verbose=1 : 디포트
# verbose=2 : 프로그래스바 삭제
# verbose=3이상 나머지 : 에포만 나온다.


#4. 평가, 예측
print('+++++++++++++++++++++++++++++++++++++++')
loss = model.evaluate(x_test, y_test)
results = model.predict([17])
print('로스 :', loss)
print('[17]의 예측값 :', results)

'''