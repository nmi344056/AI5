import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# GPU에서 사용할 
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)    # torch : 2.4.1+cu124 사용 DEVICE : cuda

#1. 데이터
x = np.array(range(100))
y = np.array(range(1, 101))
x_predict = np.array([101, 102])

# [실습] train_test_split 사용해서 만들기 / 예측값 : [101, 102]

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# torch.Size([7, 1]) torch.Size([7, 1]) torch.Size([3, 1]) torch.Size([3, 1])

#2. 모델 구성
model = nn.Sequential(
    nn.Linear(1, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 2),
    nn.Linear(2, 1),
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
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss : {}'.format(epoch, loss))   # verbose

#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()            # 평가 모드, 가중치와 기울기 갱신 X

    with torch.no_grad():   # ~~~~~~~~~~
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print('최종 loss :', loss2)

results = model(torch.Tensor(x_predict).to(DEVICE))    # .to(DEVICE) 추가
print('[12,13,14]의 예측값 :', results.detach().cpu().numpy())

'''
최종 loss : 6.063298192519884e-13
[12,13,14]의 예측값 : [[12.      ]
 [13.      ]
 [14.000001]]

'''
