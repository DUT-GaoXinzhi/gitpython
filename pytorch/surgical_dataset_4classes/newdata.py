import cv2
import torch
import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, train_root, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        self.train_root =train_root
        class_list = ["forceps1/", "scissors1/", "scissors2/", "tweezers/"]
        label = -1
        for name in class_list:
            label += 1
            data_path = self.train_root + name
            for img in os.listdir(data_path):
                img_path = data_path + img
                imgs.append((img_path, label))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        #这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = cv2.imread(fn)
        if self.transform is not None:
            img = self.transform(img)  # 是否进行transform
        return img, label# return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)



if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # 根据自己定义的那个勒MyDataset来创建数据集！注意是数据集！而不是loader迭代器
    train_data = MyDataset(r"F:/gitpython/pytorch/surgical_dataset_4classes/train/",transform)
    train_load  = torch.utils.data.DataLoader(train_data, batch_size=3,shuffle=True, num_workers=2)
    class_list = ["forceps1/", "scissors1/", "scissors2/", "tweezers/"]
    # 获取随机数据
    dataiter = iter(train_load)
    images, labels = dataiter.next()
    images = torchvision.utils.make_grid(images)
    images = images / 2 + 0.5  # unnormalize
    npimg = images.numpy()# transform to numpy
    npimg = np.transpose(npimg,(1,2,0))# resize ndarray to chw
    plt.figure(1)
    plt.title("get photo")
    plt.imshow(npimg)
    plt.show()# must!


