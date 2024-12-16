import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# GPU에서 사용할 
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)    # torch : 2.4.1+cu124 사용 DEVICE : cuda

#1. 데이터
path = "C:\\ai5\\_data\\dacon\\diabetes\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.info())     # 결측치가 없다
# print(test_csv.info())      # 결측치가 없다

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']
print(x.shape, y.shape)       # (652, 8) (652,)

x = x.to_numpy()
x = x/255.
# x = x.reshape(652, 8, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=369,
                                                    stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(type(x_train), type(y_train))
# torch.Size([521, 8]) torch.Size([521, 1]) torch.Size([131, 8]) torch.Size([131, 1])
# <class 'torch.Tensor'> <class 'torch.Tensor'>

#2. 모델 구성
# model = nn.Sequential(
#     nn.Linear(8, 64),
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

model = Model(8, 1).to(DEVICE)

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
최종 loss : 28.66108512878418
accuracy_score : 0.7176
'''
