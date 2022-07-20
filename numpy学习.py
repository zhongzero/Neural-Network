# https://blog.csdn.net/qq_40195360/article/details/84540814

#test
import numpy as np
np.random.seed(1234)
X=np.random.rand(2,3)
# X=np.array([[-1,3],[1,1]])
X=np.array([[1,2],[2,2],[1,1]])
Y=np.array([[2,2],[1,1]])
# X=np.maximum(X,3)
# X= np.zeros(4)
# X=Y*(X>0)
print(X)
print(X@np.transpose(X))


# 创建数组
X=np.array([[1,2],[3,4],[5,6]]) #传入具体数据
X=np.zeros((3,4)) #生成3x4的矩阵,初始值都为0
X=np.ones((3,4)) #生成3x4的矩阵,初始值都为1
X=np.eye(3) #生成3x3的单位矩阵
np.random.seed(1234) #赋予随机种子
X=np.random.rand(3,4) #生成匀分布的3x4的随机矩阵,随机数范围为[0,1)
# 输出
print(X) #输出整个数组
print(X.shape) #输出X的大小(几行几列)
# 常用函数
# 1.一元通用函数
np.sqrt(X)    # 计算数组各元素的平方根
np.abs(X)/np.fabs(X)/np.exp(X) # ...
# 2.二元通用函数
np.add(X,Y) /X+Y     #数组的对应元素相加
np.add(X,1) / X+1    #数组的每个元素+1
np.subtract(X,Y)/X-Y #数组的对应元素相减
np.multiply(X,Y)/X*Y #数组的对应元素相乘
#...
np.matmul(X,Y) / X@Y   #矩乘
np.maximum(X,Y) #数组的对应元素取max
np.maximum(X,0) #数组的每个元素和0取max
# 3.统计函数
np.sum(X)    #求和
np.std(X)/np.var(X)  #标准差/方差
# 4.线性代数相关
np.linalg.det(X) # 求行列式
np.transpose(X)  # 求矩阵的转置