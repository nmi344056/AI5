# import autokeras as ak      # ModuleNotFoundError: No module named 'tensorflow.keras.layers.experimental'

import autokeras as ak
import tensorflow as tf
import keras
print(ak.__version__)       # 1.0.20
print(tf.__version__)       # 2.10.1
print(keras.__version__)    # 2.10.0

import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # 이렇게도 사용 가능
print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)

#2. 모델 구성
model = ak.ImageClassifier(
    overwrite=False,
    max_trials=3,
)

#3. 컴파일, 훈련
start_time = time.time
model.fit(x_train, y_train, epochs=10, validation_split=0.15)
end_time = time.time()

##### 최적의 출력 모델 #####
best_model = model.export_model()
print(best_model.summary())

##### 최적의 모델 저장 #####
path = 'C:\\ai5\\_save\\autokeras\\'
best_model.save(path + 'keras70_autokeras1.h5')

#4. 평가, 예측
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)
print('model result :', results)

y_predict2 = best_model.predict(x_test)
# results2 = best_model.evaluate(x_test, y_test)    # best_model엔 evaluate가 없다
# print('best_model result :', results2)

print('time :', round(end_time - start_time, 3), '초')

'''
model result : [0.0345553457736969, 0.9896000027656555]

'''
