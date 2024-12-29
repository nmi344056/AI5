# 14_2 copy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)    # torch : 2.4.1+cu124 사용 DEVICE : cuda

##### 정규화 적용 #####
import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5,), (0.5,))])   # 0.5는 ((평균), (표준편차))

#  MinMax(x_train) - 평균(0.5) (고정)
# ------------------------------        = Z Score Normalization (정규화와 표준화의 짬뽕) -> -1 ~ 1
#     표준편차 (0.5) (고정)

#1. 데이터
path = 'C:/ai5/_data/torch/'
# train_dataset = MNIST(path, train=True, download=False)
# test_dataset = MNIST(path, train=False, download=False)

train_dataset = MNIST(path, train=True, download=True, transform=transf)
test_dataset = MNIST(path, train=False, download=True, transform=transf)

print(train_dataset[0][0])
# print(train_dataset[0][0].size)   # <built-in method size of Tensor object at 0x000001EFD02EB900>
print(train_dataset[0][0].shape)    # tr.Resize(30) -> torch.Size([1, 30, 30])
                                    # tr.Resize(110) -> torch.Size([1, 110, 110])
print(train_dataset[0][1])          # 5

##### 정규화(MinMax) /244. #####
# x_train, y_train = train_dataset.data/255., train_dataset.targets   # 실행 시 shape가 다시 28로 롤백
# x_test, y_test = test_dataset.data/255., test_dataset.targets
# print(x_train.shape, y_train.size())    # torch.Size([60000, 28, 28]) torch.Size([60000])

# x_train/127.5 -1 의 값의 범위는? -> -1 ~ 1 -> 정규화보다 표준화에 가깝다. -> Z Score-정규화

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

########## 잘 받아졌는지 확인 ##########
# bbb = iter(train_loader)
# aaa = next(bbb)
# print(aaa)
# print(aaa[0].shape)         # torch.Size([32, 1, 56, 56])
# print(len(train_loader))    # 1875 (60000 / 32)

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
        self.output_layer = nn.Linear(in_features=16, out_features=10)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = x.view(x.shape[0], -1)      # keras의 x = flatten() 와 같다
        x = self.hidden_layer4(x)
        x = self.output_layer(x)

        return x

model = CNN(1).to(DEVICE)

# model.summary()
# AttributeError: 'CNN' object has no attribute 'summary'
print(model)
'''
CNN(
  (hidden_layer1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.5, inplace=False)
  )
  (hidden_layer2): Sequential(
    (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.5, inplace=False)
  )
  (hidden_layer3): Sequential(
    (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.5, inplace=False)
  )
  (hidden_layer4): Linear(in_features=400, out_features=16, bias=True)
  (output_layer): Linear(in_features=16, out_features=10, bias=True)
)
'''

from torchsummary import summary
# summary(model)
# TypeError: summary() missing 1 required positional argument: 'input_size'
# summary(model, (28,28,1))   # 가로, 세로, 채널
# RuntimeError: Given groups=1, weight of size [64, 1, 3, 3], expected input[2, 28, 28, 1] to have 1 channels, but got 28 channels instead
# summary(model, (1,28,28))     # 채널, 가로, 세로
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x16 and 400x16)
summary(model, (1,56,56))       # 증폭
'''
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 54, 54]             640
              ReLU-2           [-1, 64, 54, 54]               0
         MaxPool2d-3           [-1, 64, 27, 27]               0
           Dropout-4           [-1, 64, 27, 27]               0
            Conv2d-5           [-1, 32, 25, 25]          18,464
              ReLU-6           [-1, 32, 25, 25]               0
         MaxPool2d-7           [-1, 32, 12, 12]               0
           Dropout-8           [-1, 32, 12, 12]               0
            Conv2d-9           [-1, 16, 10, 10]           4,624
             ReLU-10           [-1, 16, 10, 10]               0
        MaxPool2d-11             [-1, 16, 5, 5]               0
          Dropout-12             [-1, 16, 5, 5]               0
           Linear-13                   [-1, 16]           6,416
           Linear-14                   [-1, 10]             170
================================================================
Total params: 30,314
Trainable params: 30,314
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.97
Params size (MB): 0.12
Estimated Total Size (MB): 4.09
'''
