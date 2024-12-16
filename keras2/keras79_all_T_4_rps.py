import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import time

from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet201, DenseNet121, DenseNet169
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetMobile, NASNetLarge
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.applications import Xception

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

np_path = "C:\\ai5\\_data\\_save_npy\\keras45\\"
x_train = np.load(np_path + 'keras45_03_rps_x_train.npy')
y_train = np.load(np_path + 'keras45_03_rps_y_train.npy')

model_list = [
    (VGG19, (100, 100, 3)), 
    (ResNet50, (100, 100, 3)), 
    (ResNet101, (100, 100, 3)), 
    (DenseNet121, (100, 100, 3)),
    (MobileNetV2, (100, 100, 3)),
    (EfficientNetB0, (100, 100, 3))
]

for model_fn, model_input in model_list:
    model_fn.trainable = False

    model = Sequential()
    model.add(model_fn(include_top=False, input_shape=model_input))  # pre-trained 모델을 추가
    model.add(GlobalAveragePooling2D())
    model.add(Dense(100))
    model.add(Dense(100))
    model.add(Dense(1, activation='softmax'))

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=921)

    ##### 스케일링 #####
    x_train = x_train / 255.
    x_test = x_test / 255.
    print(np.max(x_train), np.min(x_train)) # 1.0 0.0

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=1,
                    restore_best_weights=True,
                    )

    start = time.time()

    hist =model.fit(x_train, y_train,
                    validation_split=0.2,
                    epochs=100,
                    batch_size=128, 
                    verbose=1,
                    callbacks=[es],
                    )

    end = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=1)
    print("=========================")
    print("모델명 :", model_fn.__name__, 'loss :', loss[0], 'acc :', round(loss[1],2))
    
    # 예측: sigmoid 활성화 함수 사용 시 결과가 확률로 반환되므로, 0.5 이상을 1로, 미만을 0으로 변환
    y_predict = model.predict(x_test)
    y_predict = (y_predict > 0.5).astype(int)  # 0.5 기준으로 1, 0 분류
    y_test = y_test.reshape(-1, 1)  # y_test를 2D 배열로 변환 (모델 출력과 일치시킴)

    # 정확도 계산
    acc = np.mean(y_predict == y_test)  # 정확도 계산
    print("accuracy_score : ", acc)
    print("time :", round(end-start, 2), '초')

'''
모델명 : VGG19 loss : 0.0 acc : 0.37
모델명 : ResNet50 loss : 0.0 acc : 0.3
모델명 : ResNet101 loss : 0.0 acc : 0.35
모델명 : DenseNet121 loss : 0.0 acc : 0.34
모델명 : MobileNetV2 loss : 0.0 acc : 0.34
모델명 : EfficientNetB0 loss : 0.0 acc : 0.3
'''
