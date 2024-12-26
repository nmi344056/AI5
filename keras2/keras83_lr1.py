x = 10
y = 10
w = 0.001
lr = 0.0001        # 0.1
epochs = 100

for i in range(epochs):
    hypothesis = x * w
    loss = (hypothesis - y) **2

    print('Loss ;', round(loss, 4), '\t Predict :', round(hypothesis, 4))

    up_predict = x * (w + lr)
    up_loss = (y - up_predict) **2

    down_predict = x * (w - lr)
    down_loss = (y - down_predict) **2

    if(up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr

'''
lr = 0.1
Loss ; 99.8001   Predict : 0.01
Loss ; 80.8201   Predict : 1.01
Loss ; 63.8401   Predict : 2.01
Loss ; 48.8601   Predict : 3.01
Loss ; 35.8801   Predict : 4.01
Loss ; 24.9001   Predict : 5.01
Loss ; 15.9201   Predict : 6.01
Loss ; 8.9401    Predict : 7.01
Loss ; 3.9601    Predict : 8.01
Loss ; 0.9801    Predict : 9.01
Loss ; 0.0001    Predict : 10.01
Loss ; 0.9801    Predict : 9.01
Loss ; 0.0001    Predict : 10.01    # 핑퐁 -> 갱신이 안된다 -> 1. 더하기 빼기 확인, 2. lr과 w 확인
Loss ; 0.9801    Predict : 9.01
Loss ; 0.0001    Predict : 10.01
lr가 너무 커서 핑퐁됐다.

lr = 0.001
Loss ; 99.8001   Predict : 0.01
Loss ; 99.6004   Predict : 0.02
Loss ; 99.4009   Predict : 0.03
Loss ; 99.2016   Predict : 0.04
Loss ; 99.0025   Predict : 0.05
...
Loss ; 81.7216   Predict : 0.96
Loss ; 81.5409   Predict : 0.97
Loss ; 81.3604   Predict : 0.98
Loss ; 81.1801   Predict : 0.99
Loss ; 81.0      Predict : 1.0

lr = 0.0001
Loss ; 99.8001   Predict : 0.01
Loss ; 99.7801   Predict : 0.011
Loss ; 99.7601   Predict : 0.012
Loss ; 99.7402   Predict : 0.013
Loss ; 99.7202   Predict : 0.014
...
Loss ; 97.911    Predict : 0.105
Loss ; 97.8912   Predict : 0.106
Loss ; 97.8714   Predict : 0.107
Loss ; 97.8517   Predict : 0.108
Loss ; 97.8319   Predict : 0.109
lr가 너무 작아서 갱신이 안된다.
'''
