import torch.nn.functional as F
import torch.nn as nn
import torch
import sys

sys.path.append('..')

class NNetArchitecture(nn.Module): #从torch.nn中继承nn.Module类
    #一般只需要实现__init__和forward两个函数，backward pytorch中已经写好(pytorch自动求导算梯度)
    def __init__(self, game, args):
        super(NNetArchitecture, self).__init__() #调用父类的构造函数
        #游戏所需的基本参数
        self.feat_cnt = args.feat_cnt
        self.board_x, self.board_y = game.getBoardSize() #游戏棋盘大小
        self.action_size = game.getActionSize() #合法操作数
        self.args = args

        """
            TODO: Add anything you need
        """
        self.conv1=torch.nn.Conv2d(3,10,5,padding=2)
        self.conv2=torch.nn.Conv2d(10,20,5,padding=2)
        self.conv3=torch.nn.Conv2d(20,40,5,padding=2)
        self.conv4=torch.nn.Conv2d(40,80,5,padding=2)
        self.conv5=torch.nn.Conv2d(80,10,5,padding=2)
        self.linear1=torch.nn.Linear(10*9*9,50)
        self.linear2=torch.nn.Linear(50,self.action_size)
        self.linear3_2=torch.nn.Linear(self.feat_cnt*self.board_x*self.board_y,10)
        self.linear4_2=torch.nn.Linear(10,1)

    def forward(self, s):
        # batch_size x feat_cnt x board_x x board_y
        s = s.view(-1, self.feat_cnt, self.board_x, self.board_y)   #相当于numpy中的reshape，重新定义矩阵的形状
        # print(s.shape)
        
        """
            TODO: Design your neural network architecture
            Return a probability distribution of the next play (an array of length self.action_size) 
            and the evaluation of the current state.

            pi = ...
            v = ...
        """
        # print(s.shape)
        
        pi=self.conv1(s)
        pi=F.relu(pi)
        tmp=pi
        pi=self.conv2(pi)
        pi=F.relu(pi)
        pi=self.conv3(pi)
        pi=pi+tmp
        pi=F.relu(pi)
        tmp=pi
        pi=self.conv4(pi)
        pi=F.relu(pi)
        pi=self.conv5(pi)
        pi=pi+tmp
        pi=F.relu(pi)
        # print(pi.shape)
        # pi=F.relu(pi)
        # pi=self.conv3(pi)
        # pi=F.relu(pi)
        # pi=self.conv4(pi)
        # pi=F.relu(pi)
        # pi=self.conv5(pi)
        # pi=F.relu(pi)
        # pi=self.conv6(pi)
        # pi=F.relu(pi)
        pi=pi.view(pi.size(0),-1)
        pi=self.linear1(pi)
        pi=F.relu(pi)
        pi=self.linear2(pi)
        # pi=F.relu(pi)
        # pi=self.linear3(pi)
        
        s=s.view(s.size(0),-1)
        v=self.linear3_2(s)
        v=F.relu(v)
        v=self.linear4_2(v)
        
        # print(pi.shape)
        # print(pi)
        
        # Think: What are the advantages of using log_softmax ?
        # log_softmax能够解决函数overflow和underflow，加快运算速度，提高数据稳定性
        return F.log_softmax(pi, dim=1), torch.tanh(v)