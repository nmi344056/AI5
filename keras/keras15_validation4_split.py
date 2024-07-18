import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.65, shuffle =True, random_state=133)

print(x_train, y_train)
print(x_test, y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=1))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          verbose=1,
        #   validation_data=(x_val, y_val),
          validation_split=0.3        # 30퍼센트 사용하겠다.
          )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, 
                    #   verbose=0          evaluate 에서도 verbose 가능
                    )
results = model.predict([18])
print("로스 :", loss)
print("18의 예측값 :", results)