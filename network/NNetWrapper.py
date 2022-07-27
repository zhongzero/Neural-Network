import torch.optim as optim
import torch.nn as nn
import torch
from time import time
from pytorch_classification.utils import Bar, AverageMeter
from utils import *
import os
import numpy as np
import math
import sys
sys.path.append('../../')

from network.NNetArchitecture import NNetArchitecture as nnetarch


"""
    TODO: Tune or add new arguments if you need
"""
#定义损失函数
loss_fn1=torch.nn.BCELoss(reduction='mean')
loss_fn2=torch.nn.MSELoss(reduction='mean')


args = dotdict({
    'lr': 0.003,
    'cuda': torch.cuda.is_available(),
    'feat_cnt': 3
})

class NNetWrapper():
    def __init__(self, game):
        self.nnet = nnetarch(game, args) #创建一个神经网络
        self.feat_cnt = args.feat_cnt  #特征数
        self.board_x, self.board_y = game.getBoardSize()  #棋盘大小
        self.action_size = game.getActionSize()

        """
            TODO: Choose a optimizer and scheduler

            self.optimizer = ...
            self.scheduler = ...
        """
        #创建造一个Adam参数优化器类(优化器的作用为：给反向传播得到的梯度进行一些修改优化再进行更新) (除了Adam还有SGD等，但一般没Adam好用)
        # params(必须):给定所有需要训练的参数,lr(可选):learning_rate,除此之外还有其他一些可选参数
        self.optimizer=optim.Adam(self.nnet.parameters(),lr=args.lr)
        #创建一个StepLR的scheduler(作用为:可以调整optimizer的lr),StepLR为指定的频率进行衰减，除此之外还有指数衰减ExponentialLR等
        # optimizer(必须),step_size(必须):学习率下降间隔数,gamma:学习率调整倍数(默认为0.1)
        self.scheduler=optim.lr_scheduler.StepLR(self.optimizer,step_size=100,gamma=0.1)

        if args.cuda:
            self.nnet.cuda() #确定用CPU or GPU来跑

    def loss_pi(self, outputs, targets):
        """
            TODO: Design a policy loss function  
            #策略网络的损失函数，用于给定selection中每一个儿子的概率
        """
        #用torch.nn自带损失函数计算
        # print("pi output",outputs.shape)
        # print("pi targets",targets.shape)
        # print("!!!!")
        outputs=torch.exp(outputs)
        loss_pi=loss_fn1(outputs,targets)
        
        return loss_pi

    def loss_v(self, outputs, targets):
        """
            TODO: Design a evaluation loss function  
            #估值网络的损失函数，用于代替随机走子
        """
        #用torch.nn自带损失函数计算
        
        # print("v output",outputs.shape)
        # print("v targets",targets.shape)
        # print("!!!!")
        loss_v=loss_fn2(outputs,targets)

        return loss_v

    def train(self, batches, train_steps):

        # Switch to train mode
        self.nnet.train()

        #pytorch_classification.utils.AverageMeter:一个用来记录和更新变量的工具
        data_time = AverageMeter()
        batch_time = AverageMeter()
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        end = time()

        #state_dict变量存放训练过程中需要学习的权重和偏执系数
        print(f"Current LR: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
        bar = Bar(f'Training Net', max=train_steps)
        current_step = 0
        while current_step < train_steps:  #一共训练train_step次batch
            for batch_idx, batch in enumerate(batches):
                if current_step == train_steps:
                    break
                current_step += 1
                #一次batch

                # Obtain targets from the dataset
                boards, target_pis, target_vs = batch
                #当前棋盘局面的特征张量，概率pi的目标张量，权值v的目标张量
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                data_time.update(time() - end)

                """
                    TODO: Compute output & loss
                    out_pi, out_v = ...
                    l_pi = ... 
                    l_v = ...
                """
                # 计算策略网络和估值网络的输出结果并计算损失函数
                
                # 第一步：数据的前向传播，计算预测值p_pred
                out_pi,out_v=self.nnet(boards)
                # 第二步：计算预测值p_pred与真实值的误差
                l_pi=self.loss_pi(out_pi,target_pis)
                l_v=self.loss_v(out_v,target_vs)
                
                total_loss = l_pi + l_v
                #将averagemeter的update相当于加入键值对
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))

                """
                    TODO: Compute gradient (backward) and do optimizer step
                """
                # 注:在反向传播之前，将模型的梯度归零，不然这次算出的梯度会和之前的叠加
                self.optimizer.zero_grad()
                # 第三步：反向传播误差(pytorch会自动求导算梯度)
                l_pi.backward()
                l_v.backward()
                # 第四步：在算完所有参数的梯度后，更新整个网络的参数
                self.optimizer.step()

                batch_time.update(time() - end)
                end = time()
                bar.suffix = '({step}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                    step=current_step,
                    size=train_steps,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lpi=pi_losses.avg,
                    lv=v_losses.avg,
                )
                bar.next()

        """
            TODO: do scheduler step
        """
        # 每次epoch对scheduler进行一次更新(即对optimizer的lr进行更新)
        self.scheduler.step()
        

        bar.finish()
        print()

        return pi_losses.avg, v_losses.avg

    def predict(self, board):
        # Preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda:
            board = board.contiguous().cuda()

        with torch.no_grad():
            board = board.view(self.feat_cnt, self.board_x, self.board_y)

            # Switch to eval mode(评估模式)
            self.nnet.eval()

            """
                TODO: predict pi & v
            """
            # 数据的前向传播，计算预测值p_pred
            pi,v=self.nnet(board)

            return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)

        """
            TODO: save the model (nnet, optimizer and scheduler) in the given filepath
        """
        #保存整个模型到指定位置
        # torch.save(self.nnet,filepath)
        
        #只保存模型参数到指定位置
        # torch.save(self.nnet.state_dict(),filepath)
        savefile={'model_state':self.nnet.state_dict(),
                  'optimizer_state':self.optimizer.state_dict(),
                  'scheduler_state':self.scheduler.state_dict()
                  }
        torch.save(savefile,filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))

        """
            TODO: load the model (nnet, optimizer and scheduler) from the given filepath
        """
        #从指定位置加载整个模型
        # self.nnet=torch.load(filepath)
        
        #从指定位置加载模型参数
        # self.nnet.load_state_dict(torch.load(filepath))
        state=torch.load(filepath)
        self.nnet.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.scheduler.load_state_dict(state['scheduler_state'])
