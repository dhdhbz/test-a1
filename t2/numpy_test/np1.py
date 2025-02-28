import numpy as np
from numpy.ma.extras import hstack
import random

t1=np.array(range(1,101))
t2=t1*2
t1.shape=(10,10)
t2.shape=(10,10)
print(hstack((t1,t2)))
print("-"*100)

#reshape输出表格类型
b1=np.arange(24).reshape(2,3,4)
print(b1)
print(t1.dtype)

t3=random.random()
t3=np.round(t3,2)
print(t3)