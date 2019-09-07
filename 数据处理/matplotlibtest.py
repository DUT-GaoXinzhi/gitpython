import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-1, 1, 50)
y1 = 2*x+1
y2 = x**2

plt.figure(num=1)
plt.plot(x, y1)
plt.plot(x, y2, color='red', linewidth=2, linestyle='--')
plt.xlim((-1,2))
plt.ylim((-2,3))
# label改的是xy轴的说明
plt.xlabel("i am x")
plt.ylabel("i am y")
new_ticks = np.linspace(-1,2,5)
print(new_ticks)
# ticks标记的是xy轴上的文字描述
plt.xticks((new_ticks))
plt.yticks([-1,-1.8,-1.9,1.22,3],
           ['really good','bad','normal','good','really bad'])
# gca='get current axis'得到当前的轴
ax = plt.gca()
# 得到右边和上边的轴并设置成消失掉也就是让颜色变为没有
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# 把x坐标轴用某个ticks代替也就是用下面的坐标轴代替
# 然后就可以调整xy轴的位置
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
xpo = -1
ypo = 0
# 设置坐标轴的位置通过描述data的值来进行选择当数据值为xpo的时候为
# 原点的坐标位置为(xpo,ypo)
ax.spines['bottom'].set_position(('data',xpo))
ax.spines['left'].set_position(('data',ypo))

# 设置图例
# 注意下面的线必须要有","这样才能传进去参数
plt.figure(num=2)
l1, = plt.plot(x, y1,label='up')
l2, = plt.plot(x, y2,label='down',color='red',linestyle=":")
# handles是两条线
plt.legend(handles=[l1, l2,],labels=['aaa', 'bbb'],loc='best')

# 设置标注
plt.figure(num=3)
x = np.linspace(-10,10,100)
y = np.sin(x)
x0 = 1
y0 = 2*x0+1
plt.scatter(x0,y0,s=50,color='b')
plt.plot(x,y,color='green')
plt.plot([x0,x0],[y0,0],'k--',lw=2.5)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
# 让被线覆盖住的标签能够显示出来
# for label in ax.get_xticklabels()+ax.get_yticklabels():
#     label.set_fontsize(12)
#     label.set_bbox(dict(facecolor='white',edgecolor='none',alpha=0.7))

plt.show()
