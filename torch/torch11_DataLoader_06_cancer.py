# torch09_class_06_cancer copy

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
# print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)    # torch : 2.4.1+cu124 사용 DEVICE : cuda

#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target
# print(x.shape, y.shape)     # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=369,
    stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape)     # torch.Size([398, 30]) torch.Size([398, 1])
print(x_test.shape, y_test.shape)       # torch.Size([171, 30]) torch.Size([171, 1])
print(type(x_train), type(y_train))     # <class 'torch.Tensor'> <class 'torch.Tensor'>

from torch.utils.data import TensorDataset      # x, y 합친다.
from torch.utils.data import DataLoader         # batch 정의

# 토치데이터셋 만들기 1. x와 y를 합친다.
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)
print(train_set)        # <torch.utils.data.dataset.TensorDataset object at 0x000002238F12A250>
print(type(train_set))  # <class 'torch.utils.data.dataset.TensorDataset'>
print(len(train_set))   # 398

print(train_set[0])
# (tensor([-0.2152,  2.6011, -0.2361, -0.2800, -0.2725, -0.5721, -0.7588, -0.4218,
#         -1.5523, -0.3770, -0.2189,  1.2878, -0.2728, -0.2533, -0.3813, -0.7141,
#         -0.7654, -0.4271, -0.5642, -0.2960, -0.2551,  2.5299, -0.3189, -0.3110,
#         -0.6535, -0.7142, -0.9683, -0.5690, -1.1312, -0.3984], device='cuda:0'), tensor([1.], device='cuda:0'))
print(train_set[0][0])  # 첫 번째 x
# tensor([-0.2152,  2.6011, -0.2361, -0.2800, -0.2725, -0.5721, -0.7588, -0.4218,
#         -1.5523, -0.3770, -0.2189,  1.2878, -0.2728, -0.2533, -0.3813, -0.7141,
#         -0.7654, -0.4271, -0.5642, -0.2960, -0.2551,  2.5299, -0.3189, -0.3110,
#         -0.6535, -0.7142, -0.9683, -0.5690, -1.1312, -0.3984], device='cuda:0')
print(train_set[0][1])  # 첫 번째 y, train_set[397] 까지 있다
# tensor([1.], device='cuda:0')

# 토치데이터셋 만들기 2. batch를 넣어준다.
train_loader = DataLoader(train_set, batch_size=40, shuffle=True)
test_loader = DataLoader(test_set, batch_size=40, shuffle=False)
print(len(train_loader))    # 10
print(train_loader)         # <torch.utils.data.dataloader.DataLoader object at 0x0000026E59B93E20>
# print(train_loader[0])    # Iterator라서 에러, TypeError: 'DataLoader' object is not subscriptable

print('==============================')

# 1. for 문으로 확인
# for aaa in train_loader:
#     print(aaa)
#     break

# 2. next로 확인
# bbb = iter(train_loader)
# aaa = bbb.next()  # 파이썬 3.9 까지
# print(aaa)        # AttributeError: '_SingleProcessDataLoaderIter' object has no attribute 'next'

# bbb = iter(train_loader)
# aaa = next(bbb)
# print(aaa)          # for 문과 동일하게 츨력


#2. 모델 구성
# model = nn.Sequential(
#     nn.Linear(30, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.Linear(16, 1),
#     nn.Sigmoid(),
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        # super().__init__()            # Default
        super(Model, self).__init__()   # nn.Module에 있는 ~를 상속 받아서 쓴다.
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
    
    # 순전파
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)   
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.sigmoid(x)   

        return x

model = Model(30, 1).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    # model.train()                     # 훈련모드, Default
    total_loss = 0

    for x_batch, y_batch in loader:     # batch 단위로 훈련
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch) # 여기까지 순전파

        loss.backward()                 # 기울기(gradient) 값 계산까지, 역전파 시작
        optimizer.step()                # 가중치(w) 갱신, 역전파 끝
        total_loss += loss.item()
    return total_loss / len(loader)

epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epoch: {}, loss : {}'.format(epoch, loss))   # verbose

print('==============================')

#4. 평가, 예측
def evaluate(model, criterion, loader):
    model.eval()                    # 평가 모드 (역전파, 가중치 갱신, Dropout, Batch Normalization 를 X / 기울기 갱신은 세모)
    total_loss = 0

    for x_batch, y_batch in loader:
        with torch.no_grad():           # ~
            y_predict = model(x_batch)
            loss2 = criterion(y_batch, y_predict)
            total_loss += loss2.item()
    return total_loss / len(loader)

last_loss = evaluate(model, criterion, test_loader)
print('최종 loss :', last_loss)

from sklearn.metrics import accuracy_score
y_predict = model(x_test)
accuracy = accuracy_score(y_test.cpu().numpy(), np.round(y_predict.detach().cpu().numpy()))
print('accuracy_score : {:.4f}'.format(accuracy))

print('==========')

# [실습] 밑 부분 완성 (DataLoader를 사용해서 aaccuracy_score)

# 1. GPT
all_preds = []
all_labels = []
model.eval()

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        y_predict = model(x_batch)
        # 0.5 기준으로 예측값 결정
        preds = (y_predict > 0.5).float()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# 정확도 계산
accuracy1 = accuracy_score(all_labels, all_preds)
print('accuracy_score : {:.4f}'.format(accuracy1))

print('==========')

# 2. 누리
def acc_score(model, loader):
    x_test = []
    y_test = []
    for x_batch, y_batch in loader:
        x_test.extend(x_batch.detach().cpu().numpy())
        y_test.extend(y_batch.detach().cpu().numpy())
    x_test = torch.FloatTensor(x_test).to(DEVICE)
    y_pre = model(x_test)
    acc = accuracy_score(y_test, np.round(y_pre.detach().cpu().numpy()))
    return acc

acc = acc_score(model, test_loader)
print('acc_score :', acc)

'''
최종 loss : 1.1261860311031342
accuracy_score : 0.9883
'''
