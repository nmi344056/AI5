import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# GPU에서 사용할 
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)    # torch : 2.4.1+cu124 사용 DEVICE : cuda

#1. 데이터
x = np.array([range(10)]).transpose()
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [10,9,8,7,6,5,4,3,2,1]
              ]).transpose()
print(x.shape, y.shape)     # (10, 1) (10, 3)

# [실습] 만들기 / 예측값 : [10]

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).to(DEVICE)
print(x.shape)  # torch.Size([10, 1, 1])

#2. 모델 구성
model = nn.Sequential(
    nn.Linear(1, 10),
    nn.Linear(10, 8),
    nn.Linear(8, 6),
    nn.Linear(6, 4),
    nn.Linear(4, 3),
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()

    hypothesis = model(x)

    loss = criterion(hypothesis, y)

    loss.backward()
    optimizer.step()

    return loss.item()

epochs = 2000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss : {}'.format(epoch, loss))   # verbose

#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()            # 평가 모드, 가중치와 기울기 갱신 X

    with torch.no_grad():   # ~~~~~~~~~~
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss :', loss2)

results = model(torch.Tensor([[10]]).to(DEVICE))    # .to(DEVICE) 추가
print('10의 예측값 :', results.detach().cpu().numpy())

'''
optim.SGD
최종 loss : 0.005317167844623327
10의 예측값 : [[ 1.1000000e+01  1.5733333e+00 -3.5762787e-07]]

optim.Adam
최종 loss : 0.005317163188010454
10의 예측값 : [[1.1000002e+01 1.5733333e+00 4.7683716e-07]]
'''
