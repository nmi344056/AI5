import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# GPU에서 사용할 
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)    # torch : 2.4.1+cu124 사용 DEVICE : cuda

#1. 데이터
datasets = load_iris()

x = datasets.data                         # 넘파이 형태
y = datasets.target
# print(x.shape, y.shape)                 # (150, 4) (150,) -> input=4

# x = torch.FloatTensor(x)
# y = torch.LongTensor(y)
# print(x.shape, y.shape)                 # torch.Size([150, 4]) torch.Size([150])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1004,
                                                    stratify=y)

scaler = StandardScaler()                 # scaler를 하면 넴파이 형태로 바뀐다?
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE) # 텐서 형태로 변환
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

# print(x_train.size(), y_train.size())   # torch.Size([120, 4]) torch.Size([120])
# print(x_test.size(), y_test.size())     # torch.Size([30, 4]) torch.Size([30])

#2. 모델 구성
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
    # nn.Softmax()                  # Softmax가 아닌 컴파일에서 CrossEntropyLoss 사용
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x_train, y_train):
    # model.train()                 # 훈련모드, Default
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train) # 여기까지 순전파

    loss.backward()                 # 기울기(gradient) 값 계산까지, 역전파 시작
    optimizer.step()                # 가중치(w) 갱신, 역전파 끝

    return loss.item()

EPOCHS = 1000       # 특정 수치를 대문자로 하는 경우 = 특정 상수, 고정한 상수, 값을 바꾸지 않겠다
for epoch in range(1, EPOCHS + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss : {:.8f}'.format(epoch, loss))   # verbose
    print(f'epoch: {epoch}, loss : {loss:.8f}')             # verbose, 같다

print('==============================')

#4. 평가, 예측
def evaluate(model, criterion, x_test, y_test):
    model.eval()                    # 평가 모드 (역전파, 가중치 갱신, Dropout, Batch Normalization 를 X / 기울기 계산은 세모)

    with torch.no_grad():           # 기울기 계산을 하지 않는다
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
    return loss.item()

last_loss = evaluate(model, criterion, x_test, y_test)
print('최종 loss :', last_loss)

# print(model)
'''
Sequential(
  (0): Linear(in_features=4, out_features=32, bias=True)
  (1): ReLU()
  (2): Linear(in_features=32, out_features=32, bias=True)
  (3): ReLU()
  (4): Linear(in_features=32, out_features=16, bias=True)
  (5): ReLU()
  (6): Linear(in_features=16, out_features=3, bias=True)
)
'''

# [실습] 밑 부분 완성 (x_test로 aaccuracy_score)
from sklearn.metrics import accuracy_score

y_predict = model(x_test)
# print(y_predict)
# tensor([[ -5.5989,   9.3434,  -3.7573],
#         ...
#         [ -9.8327,   5.8857,   0.2753]], device='cuda:0',

y_predict = torch.argmax(y_predict, 1)      # dim=1
# print(y_predict)
# tensor([1, 0, 0, 0, 0, 2, 1, 0, 1, 2, 1, 1, 0, 0, 0, 0, 2, 1, 2, 1, 2, 2, 0, 2,
#         2, 2, 2, 1, 1, 1], device='cuda:0')

# y_predict = np.round(y_predict.detach().cpu().numpy())
# y_test = y_test.cpu().numpy()
# accuracy = accuracy_score(y_test, y_predict)
accuracy = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())
print('accuracy_score : {:.4f}'.format(accuracy))

score = (y_predict == y_test).float().mean()    # 위와 동일
# print('accuracy_score : {:.4f}'.format(score))
print(f'accuracy_score : {score:.4f}')

'''
최종 loss : 0.30350616574287415
accuracy_score : 0.9667
accuracy_score : 0.9667
'''
