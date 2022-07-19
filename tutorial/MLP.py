import numpy as np
from matplotlib import pyplot as plt

"""
    STEP 1: Define the base class of the neural network layer.
"""

# Base class of the nnet layer
class Layer():
    def __init__(self):
        pass
    
    # Forward propagation function: compute the output by input x
    def forward(self, x):
        raise NotImplementedError
    
    # Backward propagation function: compute dE/dW and dE/dx by node_grad(dE/dy)
    def backward(self, node_grad):
        raise NotImplementedError
    
    # Update function: update the weights by gradients
    def update(self, learning_rate):
        raise NotImplementedError

"""
    STEP 2: Implement the activation functions.
"""

class Sigmoid(Layer):    
    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y
    
    def backward(self, node_grad):
        return node_grad * (self.y * (1 - self.y))
    
    def update(self, learning_rate):
        pass

class Relu():        
    def forward(self, x):
        """
            TODO: Finish the forward propagation function and save some variables if you need
        """
        self.y=np.maximum(x,0)
        return self.y

    
    def backward(self, node_grad):
        """
            TODO: Finish the backward propagation function 
        """
        return node_grad*(self.y>0)
    
    def update(self, learning_rate):
        pass

class Softmax_Cross_Entropy():
    def forward(self, x):
        """
            TODO: Finish the forward propagation function and save some variables if you need
        """
        y=np.exp(x)
        Sum=np.sum(y)
        self.y=np.divide(y,Sum) # e^xi/sum(e^xi)
        return self.y
    
    def backward(self, label):
        """
            TODO: Finish the backward propagation function 
        """
        return self.y-label
        #反向回溯把softmax函数和交叉熵函数的复合函数一起反向回溯求偏导
    
    def update(self, learning_rate):
        pass

"""
    STEP 3: Implement the linear layer.
"""

class Linear(Layer):    
    # 前一层大小为size_in,后一层大小为size_out,W为[size_in x size_out]的权值矩阵，bias为大小为[1 x size_out]的偏移值向量
    def __init__(self, size_in, size_out, with_bias):
        self.size_in = size_in
        self.size_out = size_out
        self.with_bias = with_bias
        self.W = self.initialize_weight()
        if with_bias:
            self.b = np.zeros(size_out)
    
    
    def initialize_weight(self):
        epsilon = np.sqrt(2.0 / (self.size_in + self.size_out))
        return epsilon * (np.random.rand(self.size_in, self.size_out) * 2 - 1)
        #生成权值在[-epsilon,epsilon]中且均匀分布的随机矩阵，大小为 [size_in x size_out]
        
    def forward(self, x):
        """
            TODO: Finish the forward propagation function and save some variables if you need
        """
        self.x=np.array([x])
        self.y=np.add( np.matmul( x , self.W ) ,self.b )  # y=XW+B
        return self.y
        # x为[1 x size_in]的输入向量，self.y为[1 x size_out]的输出向量
    
    def backward(self, node_grad):
        """
            TODO: Finish the backward propagation function and save gradients of W and b
        """
        self.grad_y=np.array([node_grad])
        return node_grad@np.transpose(self.W)
        # 输入大小为[1 x size_out]的向量，输出大小为[1 x size_in]的向量
    
    def update(self, learning_rate):
        """
            TODO: Update W and b by gradients calculated in the backward propagation function
        """
        # print(np.transpose(self.x))
        # print(self.grad_y)
        self.W= self.W -learning_rate * ( np.transpose(self.x) @ self.grad_y )
        #通过learning_rate,前一层的输入数据，和算出的对后一层输出数据的梯度 来算出delta(W)
        

"""
    STEP 4: Combine all parts into the MLP.
"""

class MLP():    
    def __init__(self, layer_size, with_bias=True, activation="sigmoid", learning_rate=1):
        assert len(layer_size) >= 2
        self.layer_size = layer_size
        self.with_bias = with_bias
        if activation == "sigmoid":
            self.activation = Sigmoid
        elif activation == "relu":
            self.activation = Relu
        else:
            raise Exception("activation not implemented")
        self.learning_rate = learning_rate
        self.build_model()
        
    def build_model(self):
        self.layers = []
        
        size_in = self.layer_size[0]
        for hu in self.layer_size[1:-1]:
            self.layers.append(Linear(size_in, hu, self.with_bias))
            self.layers.append(self.activation())
            size_in = hu
        #前面层用Sigmoid或Relu函数
            
        self.layers.append(Linear(size_in, self.layer_size[-1], self.with_bias))
        self.layers.append(Softmax_Cross_Entropy())
        #最后一层用softmax函数使值能转成概率，再用交叉熵函数来当loss损失函数
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, label):
        node_grad = label
        for layer in reversed(self.layers):
            node_grad = layer.backward(node_grad)
            
    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)
            
    def train(self, x, label):
        y = self.forward(x)
        self.backward(label)
        self.update(self.learning_rate)
    
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return np.argmax(x) #选择最大值的下标
    
    def loss(self, x, label):
        y = self.forward(x)
        return -np.log(y) @ label # @在numpy中可当矩阵乘法用 ( 也可以用np.matmul(a, b) )
        #此处为交叉熵公式算 算出的概率值y与标签label 的差距


"""
    STEP 5: Test
"""

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
Y = np.array([
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1]
])
#X,Y为4组训练数据

np.random.seed(1007)
EPOCH = 1000
N = X.shape[0]

mlp = MLP([2, 4, 2], learning_rate=0.1, activation="relu") #初始化MLP

loss = np.zeros(EPOCH) #开数组，初值赋为0
for epoch in range(EPOCH): #循环训练EPOCH次
    for i in range(N):
        mlp.train(X[i], Y[i])
    #每次训练4组
    
    for i in range(N):
        loss[epoch] += mlp.loss(X[i], Y[i]) 
        
    loss[epoch] /= N 
    #求数据计算得出的答案和标准答案之间 的差距 的平均值(差距用mlp.loss()函数算出)

#画图
plt.figure()
ix = np.arange(EPOCH)
plt.plot(ix, loss)
plt.savefig("relu.png")
plt.show()