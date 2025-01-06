import tensorflow as tf
print('tf.version :', tf.__version__)
print('즉시 실행 모드 :', tf.executing_eagerly())
'''
eager mode = 즉시 실행 모드 = sess.run을 제거

tf.version : 1.14.0
즉시 실행 모드 : False

tf.version : 2.7.4
즉시 실행 모드 : True
'''
# tf.compat.v1.disable_eager_execution()
print('즉시 실행 모드 :', tf.executing_eagerly())
'''
tf.version : 2.7.4
즉시 실행 모드 : True
즉시 실행 모드 : False

2.7.4 버전에서 즉시 실행 모드를 종료 -> sess.run을 사용
'''

# tf.compat.v1.enable_eager_execution()
print('즉시 실행 모드 :', tf.executing_eagerly())
'''
tf.version : 2.7.4
즉시 실행 모드 : True
즉시 실행 모드 : False
즉시 실행 모드 : True

tensorflow==1.14으로 가상환경을 맞출 수 없지만 텐서1을 사용해야 할 때 사용한다.
'''

# 즉시 실행 모드 : 텐서1의 그래프 형태의 구성 없이 자연스러운 파이썬 문법으로 실행시킨다.
# tf.compat.v1.disable_eager_execution()  # 즉시 실행 모드 종료. // 텐스플로 1.0 문법 // Default
# tf.compat.v1.enable_eager_execution()   # 즉시 실행 모드 실행. // 텐스플로 2.0 사용 가능

hello = tf.constant('hello world')
sess = tf.compat.v1.Session()             # wanning 이 귀찮아서 사용
print(sess.run(hello))

'''
가상환경    즉시 실행 모드      사용 가능
1.14.0      disable (D)        b'hello world'
1.14.0      enable             RuntimeError: The Session graph is empty.
2.7.4       disable            b'hello world'
2.7.4       enable (D)         RuntimeError: The Session graph is empty.
'''

# Tensor1은 '그래프 연산' 모드
# Tensor2은 '즉시 실행' 모드

# tf.compat.v1.enable_eager_execution()   # 즉시 실행 모드 실행 , 텐스플로 2.0의 Default

# tf.compat.v1.disable_eager_execution()  # 즉시 실행 모드 종료, 그래프 연산으로 돌아간다. Tensor1 코드를 쓸 수 있다.

# tf.executing_eagerly()                  # True면 즉시 실행 모드, Tensor2 코드만 써야 한다.
#                                         # False면 그래프 연산 모드, Tensor1 코드를 쓸 수 있다.
