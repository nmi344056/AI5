import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR100

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)    # torch : 2.4.1+cu124 사용 DEVICE : cuda

##### 정규화 적용 #####
import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5,), (0.5,))])   # 0.5는 ((평균), (표준편차))

path = 'C:/ai5/_data/torch/'
train_dataset = CIFAR100(path, train=True, download=True, transform=transf)
test_dataset = CIFAR100(path, train=False, download=True, transform=transf)

print(train_dataset[0][0])
print(train_dataset[0][0].shape)    # torch.Size([3, 56, 56])
print(train_dataset[0][1])          # 19

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # batch_size Default = 32
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)   # 통배치해도 된다

#2. 모델
class CNN(nn.Module):
    def __init__(self, num_features):           #메서드
        super(CNN, self).__init__()

        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1),       # (n, 64, 54, 54)
            # model.Conv2D(64, (3,3), stride=1, input_shape=(56, 56, 1))    # 차이점 : ~
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),                                # (n, 64, 27, 27)
            nn.Dropout(0.5),
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3), stride=1),                 # (n, 32, 25, 25)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),                                # (n, 32, 12, 12)
            nn.Dropout(0.5),
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3,3), stride=1),                 # (n, 16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),                                # (n, 16, 5, 5)
            nn.Dropout(0.5),
        )
        self.hidden_layer4 = nn.Linear(16*5*5, 16)
        self.output_layer = nn.Linear(in_features=16, out_features=100)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = x.view(x.shape[0], -1)      # keras의 x = flatten() 와 같다
        x = self.hidden_layer4(x)
        x = self.output_layer(x)

        return x
    
model = CNN(3).to(DEVICE)

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

epochs = 50
for epoch in range(1, epochs + 1):
    loss, acc = train(model, criterion, optimizer, train_loader)    # model.fit

    val_loss, val_acc = evaluate(model, criterion, test_loader)

    print('epoch : {}, loss : {:.4f}, acc : {:.3f}, val_loss : {:.4f}, val_acc : {:.3f}'.format(
    epoch, loss, acc, val_loss, val_acc))

#4. 평가, 예측
last_loss = evaluate(model, criterion, test_loader)
print('loss, acc :', last_loss)

'''
epoch : 50, loss : 2.7811, acc : 0.304, val_loss : 2.8995, val_acc : 0.287
loss, acc : (2.899508218034007, 0.28714057507987223)
'''
