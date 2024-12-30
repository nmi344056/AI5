# keras51_RNN1.py copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

random.seed(333)
np.random.seed(333)
torch.manual_seed(333)  # torch 고정
torch.cuda.manual_seed(333)

# GPU 정의
# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)    # torch : 2.4.1+cu124 사용 DEVICE : cuda

# DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
DEVICE = 'cuda:0' if torch.cuda.is_available else 'cpu'     # 2장 이상이면 cuda: 로 지정
print(DEVICE)       # cuda

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])

y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)     # (7, 3) (7,)

# x = x.reshape(7, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)              # (7, 3, 1)
# 3-D tensor with shape (batch_size, timesteps, feature).

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
print(x.shape, y.size())    # torch.Size([7, 3, 1]) torch.Size([7, 1])

from torch.utils.data import TensorDataset      # x, y 합친다.
from torch.utils.data import DataLoader         # batch 정의

train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

# # 이터레이터 형테로 데이터 확인
# aaa = iter(train_loader)
# bbb = next(aaa)         # aaa.next()
# print(bbb)
# '''
# [tensor([[[5.],
#          [6.],
#          [7.]],

#         [[6.],
#          [7.],
#          [8.]]], device='cuda:0'), tensor([[8.], [9.]], device='cuda:0')]
# '''
# print(bbb[0].size())    # torch.Size([2, 3, 1])

#2. 모델 구성
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.RNN(input_size=1,            # 피쳐 갯수
                           hidden_size=32,          # output 노드의 갯수
                        #    num_layers=1,          # Default
                           batch_first=True,        # Default = False
                           bidirectional=True       # Default = False
                           )                        # (3, N, 1) -> (N, 3, 1) -> (N, 3, 32)
        self.fc1 = nn.Linear(3*32*2, 16)              # (N, 3*32) -> (N, 16)
        self.fc2 = nn.Linear(16, 8)                 # (N, 16) -> (N, 8)
        self.fc3 = nn.Linear(8, 1)                  # (N, 8) -> (N, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # model.add(SimpleRNN(32, input_shape=(3,1)))
        # x, hidden_state = self.cell(x)
        x, h0 = self.cell(x)
        # x, _ = self.cell(x)
        x = self.relu(x)

        # x = x.reshape(-1, 3*32*2)
        # x = x.view(-1, 3*32*2)    # RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        x = x.contiguous()
        x = x.view(-1, 3*32*2)      # 곱하기 2를 명시하지 않으면 bidirectional(2배)을 인식 못함

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

model = RNN().to(DEVICE)

from torchsummary import summary
summary(model, (3, 1))
'''
        Layer (type)               Output Shape         Param #
================================================================
               RNN-1  [[-1, 3, 64], [-1, 2, 32]]               0
              ReLU-2                [-1, 3, 64]               0
            Linear-3                   [-1, 16]           1,552
            Linear-4                    [-1, 8]             136
              ReLU-5                    [-1, 8]               0
            Linear-6                    [-1, 1]               9
================================================================
Total params: 1,697
Trainable params: 1,697
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.09
Params size (MB): 0.01
Estimated Total Size (MB): 0.10

        Layer (type)               Output Shape         Param #
================================================================
               RNN-1  [[-1, 3, 64], [-1, 2, 32]]               0
              ReLU-2                [-1, 3, 64]               0
            Linear-3                   [-1, 16]           3,088
            Linear-4                    [-1, 8]             136
              ReLU-5                    [-1, 8]               0
            Linear-6                    [-1, 1]               9
================================================================
Total params: 3,233
Trainable params: 3,233
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.09
Params size (MB): 0.01
Estimated Total Size (MB): 0.10
'''

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)     # 1e-4 = 0.0001, 0이 4개

def train(model, criterion, optimizer, loader):
    # model.train()
    epoch_loss = 0

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE).float().view(-1, 1)

        optimizer.zero_grad()                   # 기울기 계산은 batch 단위로 초기화 하기때문에 for문 안에 위치
        hypothesis = model(x_batch)             # y = xw + b
        loss = criterion(hypothesis, y_batch)

        loss.backward()                         # 기울기(gradient) 계산, 역전파 시작
        optimizer.step()                        # 가중치(w) 갱신, 역전파 끝
        epoch_loss += loss.item()               # 1epoch 에 대한 ~ 누적

    return epoch_loss / len(loader)

def evaluate(model, criterion, loader):
    model.eval()                                # 평가 모드 (역전파, 가중치 갱신, Dropout, Batch Normalization 를 X / 기울기 갱신은 세모)
    epoch_loss = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE).float().view(-1, 1)

            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)

            epoch_loss += loss.item()

    return epoch_loss / len(loader)

for epoch in range(1, 1001):
    loss = train(model, criterion, optimizer, train_loader)

    if epoch % 20 == 0 :          # 20으로 나눈 나머지가 0일 때
        print('epoch : {}, loss : {:.4f}'.format(epoch, loss))

#4. 평가, 예측
x_predict = np.array([[8, 9, 10]])

def predict(model, data):
    model.eval()
    with torch.no_grad():
        data = torch.FloatTensor(data).unsqueeze(2).to(DEVICE)  # (1, 3) -> (1, 3, 1)

        y_predict = model(data)
    return y_predict.cpu().numpy()

y_predict = predict(model, x_predict)    
print('==============================')
print(y_predict)                                   # [[10.369468]]
print(y_predict[0])                                # [10.369468]
print(f'{x_predict}의 예측값 : {y_predict[0][0]}')  # [[ 8  9 10]]의 예측값 : 10.369467735290527

'''
bidirectional
[[ 8  9 10]]의 예측값 : 10.485726356506348

'''
