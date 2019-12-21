import torch
import torch.nn as nn
from test import SimpleNet
import os
import numpy as np
import cv2
import torchvision.transforms as transforms
import torchvision
from newdata import MyDataset
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # 根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
    test_data = MyDataset(r"E:/gaoxinzhi/pytorch/surgical_dataset_4classes/test/", transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=True, num_workers=2)
    class_list = ["forceps1", "scissors1", "scissors2", "tweezers"]
    PATH = r'./shit_net.pth'
    net = SimpleNet().to(device)
    net.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0
    with open(r"./result.csv", 'w') as f:
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for i in range(4):
                    f.write(class_list[labels[i]] + "\t" + class_list[predicted[i]] + '\n')
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
