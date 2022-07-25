import torch
from glob import glob
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import numpy as np
import os
from tensorboardX import SummaryWriter

from train.Arena import Arena
from game.Players import RandomPlayer, NNPlayer

class Coach:
    def __init__(self, game, nnet, args):
        
        #生成种子
        np.random.seed()

        #游戏棋盘数据
        self.game = game
        #神经网络
        self.nnet = nnet
        #创建一个空的pnet，用于之后在其中load进10轮训练之前的数据，来查看和当前训练完的AI的对战情况
        self.pnet = self.nnet.__class__(self.game)  #__class__是用来提取type的
        #一般通用数据记录
        self.args = args

        #查看checkpoint目录下的文件数，以此来判断已经训练了多少次
        networks = sorted(glob(self.args.checkpoint+'/*'))
        #接着从上次训练结束处开始训练
        self.args.start_iter = len(networks)
        #若是从没训练过，创建一个空训练数据当作初始训练数据开始训练
        if self.args.start_iter == 0:
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='iteration-0000.pkl')
            self.args.start_iter = 1
        #载入初始训练数据
        self.nnet.load_checkpoint(
            folder=self.args.checkpoint, filename=f'iteration-{(self.args.start_iter-1):04d}.pkl') 
            #字符串前面加f表示支持python表达式
        #创建tensorboardX.SummaryWriter(tensorboardX是一个视觉模块，用于显示记录图表)
        if self.args.run_name != '':
            self.writer = SummaryWriter(log_dir='runs/'+self.args.run_name)
        else:
            self.writer = SummaryWriter()

    def learn(self):
        for i in range(self.args.start_iter, self.args.num_iters + 1): #训练50轮，且第i次训练训练数据是第j个到第i个(j在train()中算出)
            print(f'------ITER {i}------')
            #训练
            self.train(i)
            #查看和随机AI打的胜率
            if self.args.compare_with_random and i % self.args.random_compare_freq == 0:
                self.compareToRandom(i)
            #查看和自己10轮训练前AI打的胜率
            if self.args.compare_with_past and i % self.args.past_compare_freq == 0:
                self.compareToPast(i)
            print()
        self.writer.close()

    def train(self, iteration):
        datasets = []
        currentHistorySize = min(max(4, (iteration + 4)//2),self.args.num_iters_for_train_examples_history)
        #把一部分数据添加到dataset中，且数据为从max(1, iteration - currentHistorySize)到iteration
        for i in range(max(1, iteration - currentHistorySize), iteration + 1): 
            data_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-data.pkl')
            policy_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-policy.pkl')
            value_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-value.pkl')
            datasets.append(TensorDataset(data_tensor, policy_tensor, value_tensor))

        #更换类型方便操作
        dataset = ConcatDataset(datasets)
        #更换类型方便操作
        dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                num_workers=self.args.workers, pin_memory=True)

        #计算总共所需训练batch的组数
        train_steps = min(self.args.train_steps_per_iteration, 
            2 * (iteration + 1 - max(1, iteration - currentHistorySize)) * self.args.max_sample_num // self.args.train_batch_size)
        #开始训练
        l_pi, l_v = self.nnet.train(dataloader, train_steps)
        #在tensorboardX.SummaryWriter类中保存数据，方便之后可视化
        self.writer.add_scalar('loss/policy', l_pi, iteration)
        self.writer.add_scalar('loss/value', l_v, iteration)
        self.writer.add_scalar('loss/total', l_pi + l_v, iteration)
        #存储该次训练完的数据
        self.nnet.save_checkpoint(
            folder=self.args.checkpoint, filename=f'iteration-{iteration:04d}.pkl')

        del dataloader
        del dataset
        del datasets
    
    def compareToPast(self, iteration):
        #计算和自己10轮训练前AI打的胜率
        past = max(0, iteration - 10)
        self.pnet.load_checkpoint(folder=self.args.checkpoint,
                                  filename=f'iteration-{past:04d}.pkl')
        print(f'PITTING AGAINST ITERATION {past}')
        pplayer = NNPlayer(self.game, self.pnet, self.args.arena_temp)
        nplayer = NNPlayer(self.game, self.nnet, self.args.arena_temp)

        arena = Arena(nplayer.play, pplayer.play, self.game)
        nwins, pwins, draws = arena.playGames(self.args.arena_compare)

        print(f'NEW/PAST WINS : {nwins} / {pwins} ; DRAWS : {draws}\n')
        self.writer.add_scalar(
            'win_rate/to past', float(nwins + 0.5 * draws) / (pwins + nwins + draws), iteration)

    def compareToRandom(self, iteration):
        #查看和随机AI打的胜率
        r = RandomPlayer(self.game)
        nnplayer = NNPlayer(self.game, self.nnet, self.args.arena_temp)
        print('PITTING AGAINST RANDOM')

        arena = Arena(nnplayer.play, r.play, self.game)
        nwins, pwins, draws = arena.playGames(self.args.arena_compare_random)

        print(f'NEW/RANDOM WINS : {nwins} / {pwins} ; DRAWS : {draws}\n')
        self.writer.add_scalar(
            'win_rate/to random', float(nwins + 0.5 * draws) / (pwins + nwins + draws), iteration)