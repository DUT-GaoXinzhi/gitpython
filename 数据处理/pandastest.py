import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
s = pd.Series([1, 2, 3, 6, np.nan, 44, 1])
dates = pd.date_range('2019-01-01', periods=6)
# 行的索引是dates列的索引就是abcd
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
# df = pd.DataFrame(np.arange(12).reshape(3,4))
print(df)
# DateFrame的参数可以是字典的形式
# 运算数字形式的数据
df.describe()
# 进行排序每行进行排序
df.sort_index(axis=1, ascending=False)
# 按照单列的值进行排序
df.sort_values(by='a')
# 选取特定的两列纯标签进行筛选
df.loc[:,['a','b']]
# 纯数字筛选
df.iloc[3:5,1:2]
# 混合筛选
df.ix[:3, ['a', 'b']]
# 筛选出df中a列大于0.5的每一行
df[df.a>0.5]
# 添加一个新的列
df['e'] = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20190101', periods=6))
# 处理没有数据的位置
# 假设如下数据丢失
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
print(df)
# 把丢失数据的那一行丢掉或者通过改变axis丢掉列
# 默认下是how=any是整行丢掉如果是all的话是一行或者一列的数全部为null的时候丢掉
df.dropna(axis=0, how='any')
# 判断是不是丢失了数据
np.any(df.isnull())==True
# 读取文件
path = "gaoxinzhi.csv"
data = pd.read_csv(path)
# data.to_pickle('gaoxinzhi.pickle')
# concatenating是dataframe合并
df1 = pd.DataFrame(np.ones((3, 4))*0, columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.ones((3, 4))*1, columns=['a', 'b', 'c', 'd'])
df3 = pd.DataFrame(np.ones((3, 4))*2, columns=['a', 'b', 'c', 'd'])
# ignore_index可以将索引重新进行编号
res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
# join 功能可以将不同的索引更好的处理
df4 = pd.DataFrame(np.ones((3, 4))*0, columns=['a', 'b', 'c', 'f'], index=[1, 2, 3])
df5 = pd.DataFrame(np.ones((3, 4))*1, columns=['e', 'b', 'c', 'd'], index=[2, 3, 4])
# inner只是寻找相同的部分进行合并outer是合并之后将空白部分填充
res1 = pd.concat([df4, df5], join='inner',ignore_index=True)
# 横向合并的时候按照df4的索引进行标号
res2 = pd.concat([df4, df5], axis=1, join_axes=[df4.index])
# 添加一个行
s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'f'])
res3 = df4.append(s1, ignore_index=True)
# merge合并
res4 = pd.merge(df4, df5, right_index=True, left_index=True,suffixes=['df4', 'df5'], how='inner')
# plt画图
# img = pd.Series(np.random.randn(1000), index=np.arange(1000))
# img = img.cumsum()
img = pd.DataFrame(np.random.randn(1000, 4), index=np.arange(1000),columns=list("ABCD"))
img = img.cumsum()
# img.plot()
ax = img.plot.scatter(x='A', y='B', color='Blue', label='Class1')
img.plot.scatter(x='A', y='C', color='Green', label='Class2', ax=ax)
plt.show()
# 逐行遍历pandas的数据元素
for index, row in data.iterrows():
# row为返回的值
#重新设置索引的值
df_new = df.reset_index(drop=True) 


