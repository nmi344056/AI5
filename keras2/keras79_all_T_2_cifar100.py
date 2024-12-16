import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
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

model_list = [
    (VGG19, (32, 32, 3)), 
    # (Xception, (71, 71, 3)),        # Xception은 최소 71x71 크기 요구
    (ResNet50, (32, 32, 3)), 
    (ResNet101, (32, 32, 3)), 
    # (InceptionV3, (75, 75, 3)),     # InceptionV3는 최소 75x75 크기 요구
    # (InceptionResNetV2, (75, 75, 3)),
    (DenseNet121, (32, 32, 3)),
    (MobileNetV2, (32, 32, 3)),
    # (NASNetMobile, (224, 224, 3)),  # NASNetMobile은 (224, 224, 3) 입력 크기 요구
    (EfficientNetB0, (32, 32, 3))
]

for model_fn, model_input in model_list:
    model_fn.trainable = False

    model = Sequential()
    model.add(model_fn(include_top=False, input_shape=model_input))  # pre-trained 모델을 추가
    model.add(GlobalAveragePooling2D())
    model.add(Dense(100))
    model.add(Dense(100))
    model.add(Dense(100, activation='softmax'))
    # model.summary()

    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    ##### 스케일링 #####
    x_train = x_train / 255.
    x_test = x_test / 255.
    print(np.max(x_train), np.min(x_train)) # 1.0 0.0

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

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
    
    y_predict = model.predict(x_test)
    y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
    y_test = np.argmax(y_test, axis=1).reshape(-1,1)

'''
모델명 : VGG19 loss : 2.768756151199341 acc : 0.31
모델명 : ResNet50 loss : 2.2351925373077393 acc : 0.45
모델명 : ResNet101 loss : 2.679520606994629 acc : 0.33
모델명 : DenseNet121 loss : 1.9351173639297485 acc : 0.49
모델명 : MobileNetV2 loss : 3.521469831466675 acc : 0.37
모델명 : EfficientNetB0 loss : 6.0015482902526855 acc : 0.01
'''
