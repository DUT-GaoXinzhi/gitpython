import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib import animation
plt.figure(num=1)
x = np.linspace(-1, 1, 50)
y1 = 2*x+1
y2 = x**2
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



# 散点图的部分
plt.figure(num=4)
n= 1024
x = np.random.normal(0, 1, n)
y = np.random.normal(0, 1, n)
# 更改散点的颜色
T = np.add(x, y)
plt.scatter(x, y, c=T, s=75, alpha=0.5,)
plt.xlim((-1.5, 1.5))
plt.ylim((-1.5, 1.5))
# 可以将坐标轴进行隐藏
plt.xticks([])
plt.yticks([])


# 柱状图部分
plt.figure(num=5)
n = 12
x = np.arange(n)

y1 = (1 - x/float(n))*np.random.uniform(0.5, 1, n)
y2 = (1 - x/float(n))*np.random.uniform(0.5, 1, n)
Y1 = zip(x, y1)
Y2 = zip(x, y2)
plt.bar(x, y1, facecolor = "#9999ff",edgecolor="white")
plt.bar(x, -y2, facecolor = "#ff9999",edgecolor="white")
for x, y2 in Y2:
    plt.text(x, -y2-0.05, '%.2f'%(-y2), ha='center', va='top')
for x, y1 in Y1:
    # ha: horizontal alignment
    plt.text(x, y1+0.05, '%.2f'%y1, ha='center', va='bottom')
plt.grid(True)
plt.xlim((-5, n))
plt.ylim((-1.25, 1.25))
# 可以将坐标轴进行隐藏
plt.xticks([])
plt.yticks([])



# 等高线的图
plt.figure(num = 6)
def f(x,y):
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)
n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
# 将线和值进行绑定
X, Y = np.meshgrid(x,y)
# cmap是color map可以根据值在颜色图中找到对应的颜色
# 使用plt.contourf去填充contour(轮廓)
plt.contourf(X, Y, f(X, Y), 8, alpha=0.75, cmap=plt.cm.hot)
# 添加的是轮廓线
C = plt.contour(X, Y, f(X, Y), 8, colors='black')
# 在线上添加数字
plt.clabel(C, inline=True,fontsize=10.5)
plt.xticks(())
plt.yticks(())


# 传入图片
plt.figure(num=7)
# 用像素点模拟图片
a = np.array([0.31513153135,0.1564651254,0.51678534535,
              0.15646765444,0.1646524525,0.45534545345,
              0.48641513213,0.15646557613,0.6649879843]).reshape(3, 3)
# interpolation的参考地址是http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
plt.imshow(a,interpolation='nearest', cmap='bone', origin='lower')
plt.xticks([])
plt.yticks([])


# 绘制3d图形
fig = plt.figure(num=8)
# 在figure上面增加一个figure背景
ax = Axes3D(fig)
x = np.arange(-4, 4, 0.25)
y = np.arange(-4, 4, 0.25)
# meshgrid可以生成矩阵网格，类似于小的时候画课程表
X, Y = np.meshgrid(x, y)

# 解释meshgrid的作用
def meshgridtest():
    x = np.arange(-4,4,1)
    y = np.arange(3,10,1)
    print(x)
    print(y)
    X,Y = np.meshgrid(x,y)
    print(X)
    print(Y)
# 如果不能理解meshgrid的作用可以执行上面的函数

R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
# rstride是跨度，cstride是另一个方向的跨度行跨和列跨
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
# zdir是在哪个方向投影
ax.contour(X,Y,Z,zdir='z',offset=-2,cmap='rainbow')
ax.set_zlim(-2, 2)


# 单框多图
plt.figure()
plt.subplot(2, 1, 1)
plt.plot([0, 1], [0, 1])
plt.subplot(2, 3, 4)
plt.plot([0, 1], [0, 2])
plt.subplot(2, 3, 5)
plt.plot([0, 1], [0, 3])
plt.subplot(2, 3, 6)
plt.plot([0, 1], [0, 4])
# 方式2subplot2grid
plt.figure()
# 创建三行三列的表格起始点是00横向跨度是3纵向跨度是1
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)
# 在方格里面添加数据线
ax1.plot([1, 2], [1, 2])
# plt全部变为set_...
ax1.set_title('set title')
# 从第二行第零列开始横向跨度是2纵向跨度是1
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=1)
ax3 = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=1, rowspan=1)
ax5 = plt.subplot2grid((3, 3), (2, 1), colspan=1, rowspan=1)
# 方式3gridspec
# gridspec有三行三列
gs = gridspec.GridSpec(3, 3)
ax1= plt.subplot(gs[0, :])
ax2= plt.subplot(gs[1, 2])
ax3= plt.subplot(gs[1:, 2])
ax4= plt.subplot(gs[-1, 0])
ax5= plt.subplot(gs[-1, -2])

# 图中图
fig = plt.figure()
x = [1, 2, 3, 4, 5, 6, 7]
y = [1, 3, 4, 2, 5, 8, 6]
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax1 = fig.add_axes([left, bottom, width, height])
ax1.plot(x, y, 'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('title')

left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(x, y, 'b')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('title  inside 1')

plt.axes([0.6, 0.2, 0.25, 0.25])
plt.plot(y[::1],x,'g')
plt.xlabel('x')
plt.ylabel('y')
plt.title(' plt inside 2')


# 次坐标轴
x = np.arange(0, 10, 0.1)
y1 = 0.05*x**2
y2 = -1*y1
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, y1,'g-')
ax2.plot(x, y2, 'b--')

ax1.set_xlabel('X data')
ax1.set_ylabel('Y1',color='g')
ax2.set_ylabel('Y2',color='b')

# 动画形式展现数据
fig, ax = plt.subplots()
x = np.arange(0,2*np.pi, 0.1)
line, =ax.plot(x, np.sin(x))
def animate(i):
    line.set_ydata(np.sin(x+i/10))
    return line,
def init():
    line.set_ydata(np.sin(x))
    return line,
# frames总共的长度也就是100帧init_func动画最开始的样子interval更新频率ms blit是更新整张图片的点还是只是更新已经改变的点
ani = animation.FuncAnimation(fig = fig,func=animate, frames=100,init_func=init,interval=20,blit=False)

plt.show()




