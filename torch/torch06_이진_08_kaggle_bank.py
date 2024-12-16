import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# GPU에서 사용할 
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)    # torch : 2.4.1+cu124 사용 DEVICE : cuda

#1. 데이터
path = "C:\\ai5\\_data\\kaggle\\playground-series-s4e1\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.isnull().sum())     # 결측치가 없다
print(test_csv.isnull().sum())

encoder = LabelEncoder()
train_csv['Geography'] = encoder.fit_transform(train_csv['Geography'])
test_csv['Geography'] = encoder.fit_transform(test_csv['Geography'])
train_csv['Gender'] = encoder.fit_transform(train_csv['Gender'])
test_csv['Gender'] = encoder.fit_transform(test_csv['Gender'])

train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

###############################################
from sklearn.preprocessing import MinMaxScaler

train_scaler = MinMaxScaler()

train_csv_copy = train_csv.copy()

train_csv_copy = train_csv_copy.drop(['Exited'], axis = 1)

train_scaler.fit(train_csv_copy)

train_csv_scaled = train_scaler.transform(train_csv_copy)

train_csv = pd.concat([pd.DataFrame(data = train_csv_scaled), train_csv['Exited']], axis = 1)

test_scaler = MinMaxScaler()

test_csv_copy = test_csv.copy()

test_scaler.fit(test_csv_copy)

test_csv_scaled = test_scaler.transform(test_csv_copy)

test_csv = pd.DataFrame(data = test_csv_scaled)
###############################################

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']
print(x.shape, y.shape)     # (165034, 10) (165034,)

# print(type(x))    # <class 'pandas.core.frame.DataFrame'>
# print(type(y))    # <class 'pandas.core.series.Series'>

x = x.to_numpy()
x = x/255.
y = y.values

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
'''
ValueError: could not determine the shape of object type 'Series'

y = y.values 추가
torch.Size([132027, 10]) torch.Size([132027, 1]) torch.Size([33007, 10]) torch.Size([33007, 1])
<class 'torch.Tensor'> <class 'torch.Tensor'>

'''

#2. 모델 구성
model = nn.Sequential(
    nn.Linear(10, 64),
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
최종 loss : 19.373855590820312
accuracy_score : 0.8658
'''
