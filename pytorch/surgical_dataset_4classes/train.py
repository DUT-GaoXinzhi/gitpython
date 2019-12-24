import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import os
from newdata import MyDataset
from torch.utils.tensorboard import SummaryWriter

# 定义神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=24, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.relu4 = nn.LeakyReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=128 * 128 * 128, out_features=84)
        self.relu5 = nn.LeakyReLU()

        self.fc2 = nn.Linear(in_features=84, out_features=24)
        self.relu6 = nn.LeakyReLU()

        self.fc3 = nn.Linear(in_features=24, out_features=4)

    def forward(self, input_nb):
        output = self.conv1(input_nb)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.pool1(output)

        output = self.conv3(output)
        output = self.relu3(output)

        # output = self.conv4(output)
        # output = self.relu4(output)

        output = self.pool2(output)
        # 特征展平
        output = output.view(-1, 128 * 128 * 128)

        output = self.fc1(output)
        output = self.relu5(output)

        output = self.fc2(output)
        output = self.relu6(output)

        output = self.fc3(output)

        return output


if __name__ == "__main__":
    writer = SummaryWriter(r"E:\gaoxinzhi\gitpython\pytorch\surgical_dataset_4classes\runs")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_list = []
    count = []
    num = 0
    PATH = r'./shit_net.pth'
    # 定义图像的转换和图像数据集
    transform = transforms.Compose([transforms.ToPILImage(),
                                    # transforms.CenterCrop(256),
                                    # transforms.Resize(512),
                                    # 旋转角度之后进行训练
                                    transforms.RandomRotation(180),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (
                                        0.5, 0.5, 0.5))])  # 根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
    train_data = MyDataset(os.getcwd() + "/train/", transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=12)

    class_list = ["forceps1", "scissors1", "scissors2", "tweezers"]
    network = SimpleNet().to(device)
    network.load_state_dict(torch.load(r'./shit_net.pth'))

    # 定义损失函数和优化函数
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    tran = transforms.ToTensor()
    # 带动量的随机梯度下降器
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    print("begin training")
    for epoch in range(17):  # 多批次循环
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            writer.add_graph(network, inputs)
            # 梯度置0
            optimizer.zero_grad()
            # 正向传播，反向传播，优化
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 打印状态信息
            running_loss += loss.item()
            if i % 20 == 19:  # 每20批次打印一次
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
                loss_list.append(running_loss / 20)
                running_loss = 0.0
                count.append(num)
                num += 1
    # 绘制损失函数和训练批次的函数图像
    plt.figure(1)
    plt.title("loss - frequency")
    plt.ylabel("loss")
    plt.xlabel("training frequency")
    plt.grid(True)
    x = np.array(count)
    y = np.array(loss_list)
    plt.plot(x, y)
    plt.show()
    print('Finished Training')
    print('Finished Training begin save mode')
    # 保存模型
    PATH = r'./shit_net.pth'
    torch.save(network.state_dict(), PATH)
    print("model save finished")
