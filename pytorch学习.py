# https://blog.csdn.net/qq_27825451/article/details/90705328
# https://blog.csdn.net/qq_27825451/article/details/90550890

"""
1.torch.nn.Module的基本属性

torch.nn.Module的基本定义(有一部分没有完全列出来)
class Module(object):
    def __init__(self):
    def forward(self, *input):

    def add_module(self, name, module):
    def cuda(self, device=None):
    def cpu(self):
    def __call__(self, *input, **kwargs):
    def parameters(self, recurse=True):
    def named_parameters(self, prefix='', recurse=True):
    def children(self):
    def named_children(self):
    def modules(self):  
    def named_modules(self, memo=None, prefix=''):
    def train(self, mode=True):
    def eval(self):
    def zero_grad(self):
    def __repr__(self):
    def __dir__(self):

在定义自已的网络的时候，需要继承nn.Module类，并重新实现构造函数__init__构造函数和forward这两个方法

一般把网络中具有可学习参数的层（如全连接层、卷积层等）放在构造函数__init__()中
一般把不具有可学习参数的层(如ReLU、dropout、BatchNormanation层)直接在forward方法里面使用nn.functional来代替(但也可放在构造函数中)
forward方法是必须要重写的，它是实现模型的功能，实现各个层之间的连接关系的核心


例1：

import torch
import torch.nn.functional as F
 
class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()  # 调用父类的构造函数
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(3, 32, 3, 1, 1)
 
        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.dense2 = torch.nn.Linear(128, 10)
 
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)     #relu不包含可学习参数，直接用torch.nn.functional中的函数来代替即可，而不用在__init__中定义一层
        x = F.max_pool2d(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
 
model = MyNet()
print(model)

运行结果为：(按构造的顺序来输出)
MyNet(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (dense1): Linear(in_features=288, out_features=128, bias=True)
  (dense2): Linear(in_features=128, out_features=10, bias=True)
)


例2：

import torch.nn as nn
from collections import OrderedDict
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv_block=nn.Sequential()
        self.conv_block.add_module("conv1",nn.Conv2d(1, 3, 2, 1, 1))
        self.conv_block.add_module("relu1",nn.ReLU())
        self.conv_block.add_module("pool1",nn.MaxPool2d(2))
 
        self.dense_block = nn.Sequential()
        self.dense_block.add_module("dense1",nn.Linear(3 * 3 * 2, 2))
        self.dense_block.add_module("relu2",nn.ReLU())
        self.dense_block.add_module("dense2",nn.Linear(2, 10))
 
    def forward(self, x):
        conv_out = self.conv_block(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense_block(res)
        return out

model = MyNet()
print("model")
print(model)
print("---------------------")
print("model.children()")
for i in model.children():  #依次给出model的所有sequential的迭代器，不带sequential的名字
    print(i)
print("---------------------")
print("model.named_children()")
for i in model.named_children():#依次给出model的所有sequential的迭代器，且带sequential的名字
    print(i)
print("---------------------")
print("model.modules()")
for i in model.modules():#依次给出model的所有module的迭代器，不带module的名字
    print(i)
print("---------------------")
print("model.named_modules()")
for i in model.named_modules():#依次给出model的所有module的迭代器，且带module的名字
    print(i)
print("---------------------")
print("model.parameters()")
for i in model.parameters():#依次给出model的所有parameter的迭代器，不带parameter的名字
    print(i)
print("---------------------")
print("model.named_parameters()")
for i in model.named_parameters():#依次给出model的所有parameter的迭代器，且带parameter的名字
    print(i)
print("---------------------")


运行结果为：
model
MyNet(
  (conv_block): Sequential(
    (conv1): Conv2d(1, 3, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
    (relu1): ReLU()
    (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (dense_block): Sequential(
    (dense1): Linear(in_features=18, out_features=2, bias=True)
    (relu2): ReLU()
    (dense2): Linear(in_features=2, out_features=10, bias=True)
  )
)
---------------------
model.children()
Sequential(
  (conv1): Conv2d(1, 3, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
Sequential(
  (dense1): Linear(in_features=18, out_features=2, bias=True)
  (relu2): ReLU()
  (dense2): Linear(in_features=2, out_features=10, bias=True)
)
---------------------
model.named_children()
('conv_block', Sequential(
  (conv1): Conv2d(1, 3, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
))
('dense_block', Sequential(
  (dense1): Linear(in_features=18, out_features=2, bias=True)
  (relu2): ReLU()
  (dense2): Linear(in_features=2, out_features=10, bias=True)
))
---------------------
model.modules()
MyNet(
  (conv_block): Sequential(
    (conv1): Conv2d(1, 3, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
    (relu1): ReLU()
    (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (dense_block): Sequential(
    (dense1): Linear(in_features=18, out_features=2, bias=True)
    (relu2): ReLU()
    (dense2): Linear(in_features=2, out_features=10, bias=True)
  )
)
Sequential(
  (conv1): Conv2d(1, 3, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
Conv2d(1, 3, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
ReLU()
MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
Sequential(
  (dense1): Linear(in_features=18, out_features=2, bias=True)
  (relu2): ReLU()
  (dense2): Linear(in_features=2, out_features=10, bias=True)
)
Linear(in_features=18, out_features=2, bias=True)
ReLU()
Linear(in_features=2, out_features=10, bias=True)
---------------------
model.named_modules()
('', MyNet(
  (conv_block): Sequential(
    (conv1): Conv2d(1, 3, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
    (relu1): ReLU()
    (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (dense_block): Sequential(
    (dense1): Linear(in_features=18, out_features=2, bias=True)
    (relu2): ReLU()
    (dense2): Linear(in_features=2, out_features=10, bias=True)
  )
))
('conv_block', Sequential(
  (conv1): Conv2d(1, 3, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
))
('conv_block.conv1', Conv2d(1, 3, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1)))
('conv_block.relu1', ReLU())
('conv_block.pool1', MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
('dense_block', Sequential(
  (dense1): Linear(in_features=18, out_features=2, bias=True)
  (relu2): ReLU()
  (dense2): Linear(in_features=2, out_features=10, bias=True)
))
('dense_block.dense1', Linear(in_features=18, out_features=2, bias=True))
('dense_block.relu2', ReLU())
('dense_block.dense2', Linear(in_features=2, out_features=10, bias=True))
---------------------
model.parameters()
Parameter containing:
tensor([[[[ 0.3384, -0.3845],
          [-0.1613,  0.3380]]],


        [[[ 0.2801, -0.4308],
          [-0.4579, -0.4079]]],


        [[[ 0.4746,  0.2603],
          [ 0.2364,  0.3468]]]], requires_grad=True)
Parameter containing:
tensor([-0.0088,  0.0850, -0.2161], requires_grad=True)
Parameter containing:
tensor([[ 0.0889, -0.1723, -0.1162,  0.1607, -0.2197,  0.2246, -0.1110,  0.1100,
          0.0349, -0.1866,  0.0958,  0.0758, -0.1890, -0.1085, -0.0107,  0.1274,
         -0.0860, -0.2008],
        [ 0.1167,  0.2232,  0.1577,  0.0863,  0.1839, -0.0344,  0.0345,  0.1662,
          0.0696,  0.0617, -0.0599,  0.0356,  0.0662, -0.0662,  0.2344, -0.1326,
          0.1135, -0.0499]], requires_grad=True)
Parameter containing:
tensor([0.1168, 0.0047], requires_grad=True)
Parameter containing:
tensor([[-0.0934,  0.3301],
        [-0.5924, -0.0535],
        [ 0.4757, -0.6286],
        [-0.5138, -0.4037],
        [ 0.4963,  0.0071],
        [ 0.3045, -0.0059],
        [ 0.1025,  0.2137],
        [ 0.0442,  0.5752],
        [ 0.0516, -0.0108],
        [ 0.3253,  0.1881]], requires_grad=True)
Parameter containing:
tensor([-0.1619, -0.4607,  0.4397, -0.4366,  0.2577,  0.2420,  0.3210, -0.3229,
        -0.0024,  0.6178], requires_grad=True)
---------------------
model.named_parameters()
('conv_block.conv1.weight', Parameter containing:
tensor([[[[ 0.3384, -0.3845],
          [-0.1613,  0.3380]]],


        [[[ 0.2801, -0.4308],
          [-0.4579, -0.4079]]],


        [[[ 0.4746,  0.2603],
          [ 0.2364,  0.3468]]]], requires_grad=True))
('conv_block.conv1.bias', Parameter containing:
tensor([-0.0088,  0.0850, -0.2161], requires_grad=True))
('dense_block.dense1.weight', Parameter containing:
tensor([[ 0.0889, -0.1723, -0.1162,  0.1607, -0.2197,  0.2246, -0.1110,  0.1100,
          0.0349, -0.1866,  0.0958,  0.0758, -0.1890, -0.1085, -0.0107,  0.1274,
         -0.0860, -0.2008],
        [ 0.1167,  0.2232,  0.1577,  0.0863,  0.1839, -0.0344,  0.0345,  0.1662,
          0.0696,  0.0617, -0.0599,  0.0356,  0.0662, -0.0662,  0.2344, -0.1326,
          0.1135, -0.0499]], requires_grad=True))
('dense_block.dense1.bias', Parameter containing:
tensor([0.1168, 0.0047], requires_grad=True))
('dense_block.dense2.weight', Parameter containing:
tensor([[-0.0934,  0.3301],
        [-0.5924, -0.0535],
        [ 0.4757, -0.6286],
        [-0.5138, -0.4037],
        [ 0.4963,  0.0071],
        [ 0.3045, -0.0059],
        [ 0.1025,  0.2137],
        [ 0.0442,  0.5752],
        [ 0.0516, -0.0108],
        [ 0.3253,  0.1881]], requires_grad=True))
('dense_block.dense2.bias', Parameter containing:
tensor([-0.1619, -0.4607,  0.4397, -0.4366,  0.2577,  0.2420,  0.3210, -0.3229,
        -0.0024,  0.6178], requires_grad=True))
---------------------


2.自定义层及构建神经网络模型

import torch
class MyLayer(torch.nn.Module):
    '''
    因为这个层实现的功能是：y=(x+bias)@weights,所以有两个参数：
    权值矩阵weights
    偏置矩阵bias
    输入 x 的维度是(in_features,)
    输出 y 的维度是(out_features,)
    weights 的维度是(in_features, out_features)
    bias 的维度是(in_fearures,)，注意不是out_features
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(MyLayer, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features)) # 由于weights是可以训练的，所以使用Parameter来定义
        #torch.nn.Parameter继承自torch.randn，其作用将一个不可训练的类型的参数转化为可训练的类型为parameter的参数，并将这个参数绑定到module里面，成为module中可训练的参数。
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(in_features))             # 由于bias是可以训练的，所以使用Parameter来定义
        else:
            self.register_parameter('bias', None)
 
    def forward(self, input):
        tmp=input+self.bias
        y=torch.matmul(tmp,self.weight)
        return y
 
N, D_in, D_out = 10, 5, 3  # 一共10组样本，输入特征为5，输出特征为3 
 
# 先定义一个模型
class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()  # 第一句话，调用父类的构造函数
        self.mylayer1 = MyLayer(D_in,D_out)
 
    def forward(self, x):
        x = self.mylayer1(x)
 
        return x
 
model = MyNet()

#创建输入、输出数据
x = torch.randn(N, D_in)  #（10，5）
y = torch.randn(N, D_out) #（10，3）
#定义损失函数
loss_fn = torch.nn.MSELoss(reduction='sum')  # \sigma((x_i-y_i)^2)
#设置学习率
learning_rate = 1e-3
#创建造一个Adam参数优化器类(优化器的作用为：给反向传播得到的梯度进行一些修改优化再进行更新) (除了Adam还有SGD等，但一般没Adam好用)
# params(必须):给定所有需要训练的参数,lr(可选):learning_rate,除此之外还有其他一些可选参数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#创建一个StepLR的scheduler(作用为:可以调整optimizer的lr),StepLR为指定的频率进行衰减，除此之外还有指数衰减ExponentialLR等
# optimizer(必须),step_size(必须):学习率下降间隔数,gamma:学习率调整倍数(默认为0.1)
# 一般以epoch为单位进行更换
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30,gamma = 0.8 )

for epoch in range(2):
    for t in range(10):
        
        # 第一步：数据的前向传播，计算预测值p_pred
        y_pred = model(x)
    
        # 第二步：计算预测值p_pred与真实值的误差
        loss = loss_fn(y_pred, y)
        print(f"第{epoch+1}个epoch 的 第 {t+1} 次训练, 损失是 {loss.item()}")
    
        # 注:在反向传播之前，将模型的梯度归零，不然这次算出的梯度会和之前的叠加
        optimizer.zero_grad()
    
        # 第三步：反向传播误差(pytorch会自动求导算梯度)
        loss.backward()
    
        # 第四步：在算完所有参数的梯度后，更新整个网络的参数
        optimizer.step()
    # 每次epoch对scheduler进行一次更新(即对optimizer的lr进行更新)
    scheduler.step()
    
运行结果为(因为有随机，所以只是某一次的输出)：
第1个epoch 的 第 1 次训练, 损失是 287.74481201171875
第1个epoch 的 第 2 次训练, 损失是 286.4709167480469
第1个epoch 的 第 3 次训练, 损失是 285.20184326171875
第1个epoch 的 第 4 次训练, 损失是 283.93768310546875
第1个epoch 的 第 5 次训练, 损失是 282.678466796875
第1个epoch 的 第 6 次训练, 损失是 281.4242248535156
第1个epoch 的 第 7 次训练, 损失是 280.1751403808594
第1个epoch 的 第 8 次训练, 损失是 278.9311828613281
第1个epoch 的 第 9 次训练, 损失是 277.6924133300781
第1个epoch 的 第 10 次训练, 损失是 276.4590148925781
第2个epoch 的 第 1 次训练, 损失是 275.23089599609375
第2个epoch 的 第 2 次训练, 损失是 274.0081787109375
第2个epoch 的 第 3 次训练, 损失是 272.7908935546875
第2个epoch 的 第 4 次训练, 损失是 271.5791015625
第2个epoch 的 第 5 次训练, 损失是 270.37286376953125
第2个epoch 的 第 6 次训练, 损失是 269.1722717285156
第2个epoch 的 第 7 次训练, 损失是 267.977294921875
第2个epoch 的 第 8 次训练, 损失是 266.7879638671875
第2个epoch 的 第 9 次训练, 损失是 265.60430908203125
第2个epoch 的 第 10 次训练, 损失是 264.4263916015625




1次epoch表示对所有训练数据进行了一次训练
在实际训练时,将所有数据分成多个batch,每次送入一部分数据
batchsize:每个batch的训练样本数量


"""

import torch.nn as nn
import torch
nllloss = nn.NLLLoss( reduction='sum')
predict = torch.Tensor([[2, 3, 1],
                        [3, 7, 9]])
label = torch.tensor([1, 2])
output=nllloss(predict, label)
print(output)