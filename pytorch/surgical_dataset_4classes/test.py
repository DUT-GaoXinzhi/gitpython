import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torchvision
class Op_net(nn.Module):
    def __init__(self):
        super(Op_net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding=1)
        self.relu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding=1)
        self.relu4 = nn.LeakyReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc = nn.Linear(in_features=128 * 128 * 24, out_features=4)

    def forward(self, input_nb):
        self.imshow(torchvision.utils.make_grid(input_nb[0][0].detach()))

        output = self.conv1(input_nb)
        output = self.relu1(output)

        self.imshow(torchvision.utils.make_grid(output[0][0].detach()))

        output = self.conv2(output)
        output = self.relu2(output)

        self.imshow(torchvision.utils.make_grid(output[0][0].detach()))

        output = self.pool1(output)

        self.imshow(torchvision.utils.make_grid(output[0][0].detach()))

        output = self.conv3(output)
        output = self.relu3(output)

        self.imshow(torchvision.utils.make_grid(output[0][0].detach()))

        output = self.conv4(output)
        output = self.relu4(output)

        self.imshow(torchvision.utils.make_grid(output[0][0].detach()))

        output = self.pool2(output)

        self.imshow(torchvision.utils.make_grid(output[0][0].detach()))

        output = output.view(-1, 128*128*24)

        output = self.fc(output)

        return output

    def imshow(self,img):
        img = img / 2 + 1  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
if __name__ == "__main__":
    img = cv.imread(r"F:\gitpython\pytorch\surgical_dataset_4classes\train\forceps1\IMG_20191014_203948.jpg")
    res = cv.resize(img, (512, 512))
    cv.imshow("s",res)
    res.resize((1,3,512,512))
    res = torch.Tensor(res)
    net = Op_net()
    output = net.forward(res)
    print(output)
