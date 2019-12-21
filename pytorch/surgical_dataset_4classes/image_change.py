import cv2 as cv
import os
train_root  = r"F:/gitpython/pytorch/surgical_dataset_4classes/train/"
test_root  = r"F:/gitpython/pytorch/surgical_dataset_4classes/test/"
class_list = ["forceps1/", "scissors1/", "scissors2/", "tweezers/"]
count = 0
for name in class_list:
    data_path = test_root+name
    # 获取输入
    for img in os.listdir(data_path):
        img_path = data_path + img
        img_now = cv.imread(img_path)
        # 修改大小并写回文件
        res = cv.resize(img_now, (512, 512))
        cv.imwrite(img_path,res)
        count += 1
        if(count%20 == 0):
            print(count)