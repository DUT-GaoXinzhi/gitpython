import numpy as np
# 矩阵的运算都可以通过axis参数改变是对于行还是列进行计算
# 创建矩阵
array = np.array([[1,2,3],[2,3,4]])
# shape和size都是属于矩阵的属性
# shape描述矩阵的形状
print(array.shape)
# size描述矩阵元素的多少
print(array.size)
# 定义矩阵里面数据的格式
a = np.array([2,12,25],dtype=int)
# 定义一个矩阵
b = np.array([[2,24,3],
              [1,5,36]])
# 定义都是0的矩阵
c = np.zeros((3,4))
# 定义都是1的矩阵
d = np.ones((5,6))
# 定义一个空的矩阵
e = np.empty((3,4))
# 创建一个有序的矩阵从10到20步长为2
f = np.arange(10,20,2)
# 重新定义矩阵的形状
f = f.reshape((1,5))
# 生成从1到10一共20段的数列会自动匹配步长
g = np.linspace(1, 10, 20)
# 矩阵的简单运算
a = np.array([10,2,3,69]).reshape((2,2))
b = np.arange(4).reshape((2,2))
# np.sin(a)可以进行简单的运算
# print(b<3)可以显示矩阵中的元素的大小
# 下面是矩阵的乘法，普通的乘法是逐个相乘
c_dot = np.dot(a,b)
# 随机生成矩阵
ran = np.random.random((2,4))
print(ran)
# 求出矩阵的最小值最大值求和类似
# axis=1是在每一行求最小值axis=0是在每一列求最小值
np.min(ran,axis=1)
# 最大最小值的索引
np.argmax(ran)
# 求平均值
np.mean(ran)
# 求中位数
np.median(ran)
# 累加类似于斐波那契数列
np.cumsum(ran)
# 累差
np.diff(ran)
# 对应每个值的位置输出两个数组分别对应是每个非零元素的行数和列数
np.nonzero(ran)
# 排序
np.sort(ran)
# 矩阵转置
np.transpose(ran)
# 矩阵的截取所有小于0.1的都会变成0.1大于0.5的都会变成0.5
np.clip(ran, 0.1, 0.5)
# 矩阵索引类似于数组的索引
# 第二行的值
ran[1]
# 取的是第一行第0列到第一列的值后面冒号部分是前闭后开
ran[1,0:1]
# flat是迭代器可以将多维矩阵变为一维
ran.flat
# numpy数组合并
first = np.array([1,1,1])
second = np.array([2,2,2])
# 上下合并
content = np.vstack((first, second))
# 左右合并
content1 = np.hstack((first, second))
# 增加first中列的维度这样可以横向合并
content2 = np.hstack((first[:, np.newaxis], second[:, np.newaxis]))
# 多个array的合并axis 确定在哪个维度进行合并
content3 = np.concatenate((first[:, np.newaxis], second[:, np.newaxis], first[:, np.newaxis]),axis=1)
# 矩阵的分割只能等距分割
test = np.arange(12).reshape(3,4)
np.split(test, 3, axis=0)
# 实现不等量分割
np.array_split(test, 3, axis=1)
# 横向分割
np.vsplit(test, 3)
# 纵向分割
np.hsplit(test, 4)
# 复制下面的b和test是同一个矩阵
b = test
# 下面的b和test会占用不同的地址空间
b = test.copy()


