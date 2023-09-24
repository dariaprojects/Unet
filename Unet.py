import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import os
import numpy as np

class Conv2D_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv2D_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channel)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn1(x)

        return x
    
class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Encoder, self).__init__()

        self.conv = Conv2D_block(in_channel, out_channel)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p
    
class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decoder, self).__init__()

        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2, padding=0)
        self.conv = Conv2D_block(out_channel*2, out_channel)

    def forward(self, inputs, skip_layer):
        x = self.up(inputs)
        x = torch.cat([x, skip_layer], axis=1)
        x = self.conv(x)

        return x

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.e1 = Encoder(3, 32)
        self.e2 = Encoder(32, 64)
        self.e3 = Encoder(64, 128)
        self.e4 = Encoder(128, 256)
        self.b = Conv2D_block(256, 512)
        self.d1 = Decoder(512, 256)
        self.d2 = Decoder(256, 128)
        self.d3 = Decoder(128, 64)
        self.d4 = Decoder(64, 32)

        self.outputs = nn.Conv2d(32, 3, kernel_size=1, padding=0)

    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        b = self.b(p4)
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        return outputs

def train(trainData, maskData, model, loss_fn, optimizer, device="cpu"):
    mean_loss = 0
    correct = 0
    num_batches = len(trainData)
    size = len(trainData.dataset)
    model.train()
    for (batch, (train, _)), (batch2, (mask, _)) in zip(enumerate(trainData), enumerate(maskData)):
        train, mask = train.to(device), mask.to(device)
        # Compute prediction error
        pred = model(train)
        loss = loss_fn(pred, mask)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss += float(loss)
        correct += (pred == mask).type(torch.float).sum().item()
        if batch % 3 == 0:
            loss, current = loss.item(), batch * len(train)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    mean_loss /= num_batches
    accuracy = 100 * correct / size
    return mean_loss, accuracy

def test(testData, maskData, model, loss_fn, device="cpu"):
    size = len(testData.dataset)
    num_batches = len(testData)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for (batch, (test, _)), (batch2, (mask, _)) in zip(enumerate(testData), enumerate(maskData)):
            test, mask = test.to(device), mask.to(device)
            pred = model(test)
            test_loss += loss_fn(pred, mask).item()
            correct += (pred == mask).type(torch.float).sum().item()
    test_loss /= num_batches
    accuracy = 100 * correct / size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")            
    return test_loss, accuracy

def plot_metrics(Loss_train, Loss_val, Acc_train, Acc_val):  
    plt.figure(figsize=(16,8))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(Acc_train)), Acc_train, label='Точность на обучении')
    plt.plot(range(len(Acc_val)), Acc_val, label='Точность на валидации')
    plt.legend(loc='lower right')
    plt.title('Точность на обучающих и валидационных данных')

    plt.subplot(1, 2, 2)
    plt.plot(range(len(Loss_train)), Loss_train, label='Потери на обучении')
    plt.plot(range(len(Loss_val)), Loss_val, label='Потери на валидации')
    plt.legend(loc='upper right')
    plt.title('Потери на обучающих и валидационных данных')
    plt.savefig('./result.png')
    plt.show()
    
#загрузка изображений для тренировки и тестирования
BATCH_SIZE = 10 # количество тренировочных изображений 
IMG_SHAPE = 224 # размерность к которой будет преведено входное изображение
 
dataTest_dir = 'D:\Datasets\CovidTest'
dataTrain_dir = 'D:\Datasets\CovidTrain'
dataMaskTrain_dir = 'D:\Datasets\CovidMaskTrain'
dataMaskTest_dir = 'D:\Datasets\CovidMaskTest'

transform = transforms.Compose([transforms.Resize((IMG_SHAPE, IMG_SHAPE)), transforms.ToTensor()])
                                                                     
train_dataset = datasets.ImageFolder(dataTrain_dir,
                               transform=transform)
test_dataset = datasets.ImageFolder(dataTest_dir,
                               transform=transform)
maskTrain_dataset = datasets.ImageFolder(dataMaskTrain_dir,
                               transform=transform)
maskTest_dataset = datasets.ImageFolder(dataMaskTest_dir,
                               transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
maskTrain_loader = torch.utils.data.DataLoader(maskTrain_dataset, batch_size=BATCH_SIZE, shuffle=False)
maskTest_loader = torch.utils.data.DataLoader(maskTest_dataset,                                  batch_size=BATCH_SIZE, shuffle=False)



#этап обучения модели
num_epochs = 10 #максимальное количество эпох
learning_rate = 1e-2 #скорость обучения 
model = Unet() #экземпляр класса сети
loss_func = torch.nn.BCEWithLogitsLoss() #определение функции потерь
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#оптимизатор

train_loss = []
acc_train = []
test_loss = []
acc_test = []

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
   device = torch.device("cpu")
   print(device)

for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    mean_loss, accuracy = train(train_loader, maskTrain_loader, model, loss_func, 
                                optimizer, device)
    train_loss.append(mean_loss)
    acc_train.append(accuracy)
    mean_loss, accuracy = test(test_loader, maskTest_loader, model, loss_func, device)
    test_loss.append(mean_loss)
    acc_test.append(accuracy)
    
torch.save(model.state_dict(), "unet1.pth")
plot_metrics(train_loss, test_loss, acc_train, acc_test)