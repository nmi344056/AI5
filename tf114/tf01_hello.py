# pip install tensorflow==1.14
# pip install protobuf==3.20
# pip install numpy==1.16

import tensorflow as tf
print(tf.__version__)   # 1.14.0

print('hello world')    # hello world

hello = tf.constant('hello world')
print(hello)            # Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session()
print(sess.run(hello))  # b'hello world'
