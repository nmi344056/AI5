import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)    # torch : 2.4.1+cu124 사용 DEVICE : cuda

path = 'C:/ai5/_data/torch/'
train_dataset = MNIST(path, train=True, download=True)
test_dataset = MNIST(path, train=False, download=True)

print(train_dataset)
'''
Dataset MNIST
    Number of datapoints: 60000
    Root location: C:/ai5/_data/torch/
    Split: Train
'''
print(type(train_dataset))      # <class 'torchvision.datasets.mnist.MNIST'>
print(train_dataset[0])         # (<PIL.Image.Image image mode=L size=28x28 at 0x1A560EA6F70>, 5)
print(train_dataset[0][0])      # <PIL.Image.Image image mode=L size=28x28 at 0x2649AF66F70>

x_train, y_train = train_dataset.data/255., train_dataset.targets
x_test, y_test = test_dataset.data/255., test_dataset.targets

print(x_train)
print(y_train)  # tensor([5, 0, 4,  ..., 5, 6, 8])

print(x_train.shape, y_train.size())    # torch.Size([60000, 28, 28]) torch.Size([60000])
print(np.min(x_train.numpy()), np.max(x_train.numpy())) # 0.0 1.0

x_train, x_test = x_train.view(-1, 28*28), x_test.reshape(-1, 784)
print(x_train.shape, x_test.size())    # torch.Size([60000, 784]) torch.Size([10000, 784])

train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dset, batch_size=32, shuffle=True)  # batch_size Default = 32
test_loader = DataLoader(test_dset, batch_size=32, shuffle=False)   # 통배치해도 된다

#2. 모델
class DNN(nn.Module):
    def __init__(self, num_features):           #메서드
        super().__init__()

        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 128),       # num_features = 784
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.output_layer = nn.Linear(32,10)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)

        return x
    
model = DNN(784).to(DEVICE)

#3. 컴파일, 훈련 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)     # 1e-4 = 0.0001, 0이 4개

def train(model, criterion, optimizer, loader): # 함수
    # model.train()
    epoch_loss = 0
    epoch_acc = 0

    for x_batch, y_batch in loader:             # batch 단위 연산 32*784
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()                   # 기울기 계산은 batch 단위로 초기화 하기때문에 for문 안에 위치
        hypothesis = model(x_batch)             # y = xw + b
        loss = criterion(hypothesis, y_batch)

        loss.backward()                         # 기울기(gradient) 계산, 역전파 시작
        optimizer.step()                        # 가중치(w) 갱신, 역전파 끝
        epoch_loss += loss.item()

        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()
        epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)

def evaluate(model, criterion, loader):
    model.eval()                                # 평가 모드 (역전파, 가중치 갱신, Dropout, Batch Normalization 를 X / 기울기 갱신은 세모)
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            hypothesis = model(x_batch)

            loss = criterion(hypothesis, y_batch)

            epoch_loss += loss.item()

            y_predict = torch.argmax(hypothesis, 1)
            acc = (y_predict == y_batch).float().mean()
            epoch_acc += acc.item()
        
        return epoch_loss / len(loader), epoch_acc / len(loader)
# keras에서 loss, acc = model.evaluate(x_test, y_test)에 해당

epochs = 5
for epoch in range(1, epochs + 1):
    loss, acc = train(model, criterion, optimizer, train_loader)    # model.fit

    val_loss, val_acc = evaluate(model, criterion, test_loader)

    print('epoch : {}, loss : {:.4f}, acc : {:.3f}, val_loss : {:.4f}, val_acc : {:.3f}'.format(
    epoch, loss, acc, val_loss, val_acc))

'''
epoch : 20, loss : 0.0517, acc : 0.985, val_loss : 0.1143, val_acc : 0.968
'''

# [실습] 결괏값 나오게하는 코드 추가

#4. 평가, 예측
last_loss = evaluate(model, criterion, test_loader)
print('loss, acc :', last_loss)

'''
loss, acc : (0.16463106802959268, 0.9537739616613419)
'''
