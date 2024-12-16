import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# x = torch.FloatTensor(x)
# print(x.shape)  # torch.Size([3])
# print(x.size()) # torch.Size([3]), shape와 같다, torch에서 많이 사용

x = torch.FloatTensor(x).unsqueeze(1)
y = torch.FloatTensor(y).unsqueeze(1)
print(x)   # (3,) -> (3, 1), 2차원 행렬 형태로 만든다

print(x.shape, y.shape)     # torch.Size([3, 1]) torch.Size([3, 1])
print(x.size(), y.size())   # torch.Size([3, 1]) torch.Size([3, 1])

#2. 모델 구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))
model = nn.Linear(1, 1)     # (인풋, 아웃풋), y = xw + b

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    model.train()           # 훈련 모드 -> 가중치 갱신, 안써도 되지만 가독성을 위해 쓴다
    optimizer.zero_grad()   # 각 배치마다 기울기를 초기화하여, 기울기 누적에 의한 문제 해결

    hypothesis = model(x)   # y = wx + b

    loss = criterion(hypothesis, y) # loss=mse(), hypothesis - y

    loss.backward()         # 기울기(gradient)값 계산까지, gradient = loss를 w로 미분, 역전파 시작
    optimizer.step()        # 가중치(w) 갱신, 역전파 끝

    return loss.item()

epochs = 2000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss : {}'.format(epoch, loss))   # verbose

print('==============================')

#4. 평가, 예측
# loss = model.evaluate(x, y)
def evaluate(model, criterion, x, y):
    model.eval()            # 평가 모드, 가중치와 기울기 갱신 X

    with torch.no_grad():   # ~~~~~~~~~~
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss :', loss2)

results = model(torch.Tensor([[4]]))
# print('4의 예측값 :', results)         # 4의 예측값 : tensor([[4.0030]], grad_fn=<AddmmBackward0>)
print('4의 예측값 :', results.item())    # 4의 예측값 : 4.00297737121582

'''
epoch: 2000, loss : 1.393963975715451e-05
==============================
최종 loss : 1.3873225725546945e-05
4의 예측값 : 3.992530107498169
'''
