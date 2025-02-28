import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
x1=np.array([1,3,5,8])
y1=np.array([3,6,7,10])
x2=np.array([2,3,5,8])
y2=np.array([6,7,9,10])

plt.title('表格绘图测试')
plt.xlabel('数据a')
plt.ylabel('数据b')

plt.plot(x1,y1,'*',ms=20,ls='dotted')
plt.plot(x2,y2,'o',ms=10,ls='-')
plt.show()

