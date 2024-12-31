import pandas as pd
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
DEVICE = 'cuda:0' if torch.cuda.is_available else 'cpu'     # 2장 이상이면 cuda: 로 지정
print(DEVICE)       # cuda

path = 'C:\\ai5\\_data\\kaggle\\netflix\\'
train_csv = pd.read_csv(path + 'train.csv')
print(train_csv)
print(train_csv.info())
print(train_csv.describe())

# import matplotlib.pyplot as plt

# # data = train_csv[:, 1:4]
# # print(data)     # pandas.errors.InvalidIndexError: (slice(None, None, None), slice(1, 4, None))
# data = train_csv.iloc[:, 1:4]
# data['종가'] = train_csv['Close']
# print(data)

# hist = data.hist()
# plt.show()

#--------------------------------------------------
# data = train_csv.iloc[:, 1:4]
# data = (data - np.min(data)) / (np.max(data) - np.min(data))
# data = pd.DataFrame(data)
# print(data.describe())
'''
             Open        High         Low
count  967.000000  967.000000  967.000000
mean     0.419602    0.429021    0.409107
std      0.304534    0.309121    0.298979
min      0.002915    0.014577    0.000000
25%      0.128280    0.134111    0.125364
50%      0.332362    0.338192    0.326531
75%      0.725948    0.734694    0.708455
max      0.994169    1.000000    0.970845
컬럼별로 정규화가 되는 것이 아닌 전체로 정규화가 됐다 - 데이터가 틀어짐
'''

########## axis=0을 넣어주면 컬럼별로 최대 최소를 구한다. ##########
# data = train_csv.iloc[:, 1:4]
# data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
# data = pd.DataFrame(data)
# print(data.describe())
'''
             Open        High         Low
count  967.000000  967.000000  967.000000
mean     0.420363    0.420574    0.421392
std      0.307221    0.313694    0.307957
min      0.000000    0.000000    0.000000
25%      0.126471    0.121302    0.129129
50%      0.332353    0.328402    0.336336
75%      0.729412    0.730769    0.729730
max      1.000000    1.000000    1.000000
'''

from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader

class Custom_Dataset(Dataset):
    def __init__(self):
        self.csv = train_csv

        self.x = self.csv.iloc[:, 1:4].values       # 시가, 고가, 저가
        # self.x = (self.x - np.min(self.x)) / (np.max(self.x) - np.min(self.x))  # 정규화
        self.x = (self.x - np.min(self.x, axis=0)) / (np.max(self.x, axis=0) - np.min(self.x, axis=0))  # 컬럼별로 정규화

        self.y = self.csv['Close'].values

    def __len__(self):
        return len(self.x) - 30
    
    def __getitem__(self, i):
        x = self.x[i:i+30]
        y = self.y[i+30]

        return x, y

aaa = Custom_Dataset()
print(aaa)              # <__main__.Custom_Dataset object at 0x0000019DFFCBEB70>
print(type(aaa))        # <class '__main__.Custom_Dataset'>

print(aaa[0])           # 총 30개, np.int64(94))
print(aaa[0][0].shape)  # (30, 3)
print(aaa[0][1])        # 94
print(len(aaa))         # 937
# print(aaa[937])       # IndexError: index 967 is out of bounds for axis 0 with size 967
print(aaa[936])         # 출력

########## x는 (937, 30, 3), y는 (937, 1) ##########

# train_loader = DataLoader(aaa, batch_size=32)

# # 이터레이터 형테로 데이터 확인
# aaa = iter(train_loader)
# bbb = next(aaa)         # aaa.next()
# print(bbb)
# print(bbb[0].size())    # torch.Size([32, 30, 3])

#2. 모델 구성

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(input_size=3,             # 피쳐 갯수
                          hidden_size=64,           # output 노드의 갯수
                          num_layers=5,             # 
                          batch_first=True,         # 
                          )
        self.fc1 = nn.Linear(in_features=30*64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x, h0):
        x, hn = self.rnn(x, h0)

        x = torch.reshape(x, (x.shape[0], -1))

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = RNN().to(DEVICE)

#3. 컴파일, 훈련
# from torch.optim import Adam
# optim = Adam(params=model.parameters(), lr=0.001)

# import tqdm

# for epoch in range(1, 201):
#     iterator = tqdm.tqdm(train_loader)
#     for x, y in iterator:
#         optim.zero_grad()

#         h0 = torch.zeros(5, x.shape[0], 64).to(DEVICE)   # (num_layers, batch_size, hidden_size) = (5, 32, 64)

#         hypothesis = model(x.type(torch.FloatTensor).to(DEVICE), h0)

#         loss = nn.MSELoss()(hypothesis, y.type(torch.FloatTensor).to(DEVICE))

#         loss.backward() 
#         optim.step()

#         iterator.set_description(f'epoch: {epoch} loss: {loss.item()}')

save_path = 'C:\\ai5\\_save\\torch\\'
# torch.save(model.state_dict(), save_path + 't22.pth')

#4. 평가, 예측

# train_loader = DataLoader(aaa, batch_size=1)

# y_predict = []
# total_loss = 0

# with torch.no_grad():
#     model.load_state_dict(torch.load(save_path + 't22.pth', map_location=DEVICE,
#                                      weights_only=True,
#                                      ))
    
#     for x_test, y_test in train_loader:
#         h0 = torch.zeros(5, x_test.shape[0], 64).to(DEVICE)     # (5, 1, 64)

#         y_pred = model(x_test.type(torch.FloatTensor).to(DEVICE), h0)
#         y_predict.append((y_pred.item()))

#         loss = nn.MSELoss()(y_pred,
#                             y_test.type(torch.FloatTensor).to(DEVICE))
#         total_loss += loss / len(train_loader)

# # print(f'y_predict: {y_predict}, \n len: {(len(y_predict))}')
# print('total_loss: ', total_loss.item())

'''
c:\Users\AI5\anaconda3\envs\torch241\Lib\site-packages\torch\nn\modules\loss.py:538: UserWarning: Using a target size (torch.Size([1])) that is different to the input size (torch.Size([1, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  return F.mse_loss(input, target, reduction=self.reduction)
total_loss:  965.3842163085938

y값 행렬 형태로

'''

##### [실습] R2 만들기 #####
from sklearn.metrics import r2_score

#4. 평가 예측
train_loader = DataLoader(aaa, batch_size=1)

y_predict = []
total_loss = 0
y_true = []

with torch.no_grad():
    model.load_state_dict(torch.load(save_path + 't22.pth', map_location=DEVICE))
    for x_test, y_test in train_loader:
        h0 = torch.zeros(5, x_test.shape[0], 64).to(DEVICE)  # (num_layers, batch_size, hidden_size) = (5, 1, 64)
        
        y_pred = model(x_test.type(torch.FloatTensor).to(DEVICE), h0)
        y_predict.append(y_pred.cpu().numpy())  # numpy 배열로 변환 후 저장
        y_true.append(y_test.cpu().numpy())  # numpy 배열로 변환 후 저장
        
        loss = nn.MSELoss()(y_pred, y_test.type(torch.FloatTensor).to(DEVICE))
        total_loss += loss / len(train_loader)

#print(f'y_predict : {y_predict}, \n shape: {y_predict.shape}')

# numpy 배열로 변환하여 r2 계산
y_predict = np.array(y_predict).flatten()
y_true = np.array(y_true).flatten()

r2 = r2_score(y_true, y_predict)
print('R2:', r2)
print('total_loss:', total_loss.item())

'''
R2: 0.9105271100997925
total_loss: 965.3842163085938
'''
