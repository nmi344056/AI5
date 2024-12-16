import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# GPU에서 사용할 
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)    # torch : 2.4.1+cu124 사용 DEVICE : cuda

#1. 데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=369,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

from torch.utils.data import TensorDataset      # x, y 합친다.
from torch.utils.data import DataLoader         # batch 정의

# 토치데이터셋 만들기 1. x와 y를 합친다.
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

# 토치데이터셋 만들기 2. batch를 넣어준다.
train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

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

        return x

model = Model(8, 1).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    # model.train()                 # 훈련모드, Default
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
    model.eval()                    # 평가 모드 (역전파,가중치 갱신, Dropout, Batch Normalization 를 X / 기울기 갱신은 세모)
    total_loss = 0

    for x_batch, y_batch in loader:
        with torch.no_grad():           # ~
            y_predict = model(x_batch)
            loss2 = criterion(y_predict, y_batch)
            total_loss += loss2.item()
    return total_loss / len(loader)

last_loss = evaluate(model, criterion, test_loader)
print('최종 loss :', last_loss)

y_predict = model(x_test).detach().cpu().numpy()
y_test = y_test.cpu().numpy()
r2 = r2_score(y_test, y_predict)
print('r2_score : {:.4f}'.format(r2))

print('==========')

# [실습] 밑 부분 완성 (DataLoader를 사용해서 aaccuracy_score)

def R2_score(model, loader):
    x_test = []
    y_test = []
    for x_batch, y_batch in loader:
        x_test.extend(x_batch.detach().cpu().numpy())
        y_test.extend(y_batch.detach().cpu().numpy())
    x_test = torch.FloatTensor(x_test).to(DEVICE)
    y_pre = model(x_test)
    acc = r2_score(y_test, y_pre.detach().cpu().numpy())
    return acc

r2 = R2_score(model, test_loader)
print('r2_score :', r2)

'''
최종 loss : 0.2647236227989197
r2_score : 0.7944
'''
