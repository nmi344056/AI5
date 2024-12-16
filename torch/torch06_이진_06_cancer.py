# torch06_logistic_regression06_cancer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# GPU에서 사용할 
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)    # torch : 2.4.1+cu124 사용 DEVICE : cuda

#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=369,
                                                    stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
# x_train = torch.DoubleTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
# x_test = torch.DoubleTensor(x_test).to(DEVICE)

y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
# y_train = torch.DoubleTensor(y_train).unsqueeze(1).to(DEVICE)
# y_train = torch.LongTensor(y_train).unsqueeze(1).to(DEVICE)
# y_train = torch.IntTensor(y_train).unsqueeze(1).to(DEVICE)

y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
# y_test = torch.DoubleTensor(y_test).unsqueeze(1).to(DEVICE)
# y_train = torch.LongTensor(y_train).unsqueeze(1).to(DEVICE)
# y_test = torch.IntTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(type(x_train), type(y_train))
'''
Float Float
torch.Size([455, 30]) torch.Size([455, 1]) torch.Size([114, 30]) torch.Size([114, 1])
<class 'torch.Tensor'> <class 'torch.Tensor'>

Double Double
RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float

'''

#2. 모델 구성
model = nn.Sequential(
    nn.Linear(30, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.Linear(16, 1),
    nn.Sigmoid(),
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    # model.train()                 # 훈련모드, Default
    optimizer.zero_grad()

    hypothesis = model(x)

    loss = criterion(hypothesis, y) # 여기까지 순전파

    loss.backward()                 # 기울기(gradient) 값 계산까지, 역전파 시작
    optimizer.step()                # 가중치(w) 갱신, 역전파 끝

    return loss.item()

epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss : {}'.format(epoch, loss))   # verbose

print('==============================')

#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()                    # 평가 모드 (역전파,가중치 갱신, Dropout, Batch Normalization 를 X / 기울기 갱신은 세모)

    with torch.no_grad():           # ~
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

last_loss = evaluate(model, criterion, x_test, y_test)
print('최종 loss :', last_loss)

# [실습] 밑 부분 완성 (x_test로 aaccuracy_score)
from sklearn.metrics import accuracy_score

# y_predict = model(torch.Tensor(x_test).to(DEVICE))
y_predict = model(x_test)
# print(y_test)       # [1.]], device='cuda:0')
# print(y_predict)    # [1.0000e+00]], device='cuda:0', grad_fn=<SigmoidBackward0>)

# y_predict = np.round(y_predict.detach().cpu().numpy())
# y_test = y_test.cpu().numpy()
# accuracy = accuracy_score(y_test, y_predict)
accuracy = accuracy_score(y_test.cpu().numpy(), np.round(y_predict.detach().cpu().numpy()))
print('accuracy_score : {:.4f}'.format(accuracy))

'''
최종 loss : 1.2006281614303589
accuracy_score : 0.9912280701754386
'''
