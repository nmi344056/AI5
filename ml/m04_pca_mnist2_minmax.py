import numpy as np
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)

x = np.concatenate([x_train, x_test], axis=0)
print(x.shape)                          # (70000, 28, 28)

# 스케일링
x = x/255.
print(np.max(x), np.min(x))             # 1.0 0.0

x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
print(x.shape)                          # (70000, 784)

pca = PCA(n_components=28*28)
x = pca.fit_transform(x)

evr_cumsum = np.cumsum(pca.explained_variance_ratio_)
# print(evr_cumsum)

print('0.95 :', np.argmax(evr_cumsum>=0.95)+1)      # 154
print('0.99 :', np.argmax(evr_cumsum>=0.99)+1)      # 331
print('0.999 :', np.argmax(evr_cumsum>=0.999)+1)    # 486
print('1.0 :', np.argmax(evr_cumsum>=1.0)+1)        # 713
