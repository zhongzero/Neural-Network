from torch import multiprocessing as mp

from train.Coach import Coach
from network.NNetWrapper import NNetWrapper as nn
from utils import *
from libcpp import Game

args = dotdict({
    'run_name': 'Gomoku',
    'workers': mp.cpu_count() - 1,
    #载入start_iter-1完成的训练数据，从start_iter开始依次训练
    'start_iter': 1,
    #训练到num_iters结束
    'num_iters': 50,
    #一个batch的训练数据大小
    'train_batch_size': 512,
    'train_steps_per_iteration': 200,
    'max_sample_num': 10000, 
    'num_iters_for_train_examples_history': 100,
    'temp_threshold': 10,
    'temp': 1,
    'arena_compare_random': 50,
    'arena_compare': 50,
    'arena_temp': 0.1,
    'compare_with_random': True,
    'random_compare_freq': 10,
    'compare_with_past': True,
    'past_compare_freq': 10,
    #存储训练完的数据的文件夹
    'checkpoint': 'checkpoint',
    #数据存储地址
    'data': '../Neural_Network_data',
})

if __name__ == "__main__":
    # Create a Gomoku 9x9 game instance
    g = Game(9, 5)

    # Create a neural network instance
    nnet = nn(g)

    c = Coach(g, nnet, args)
    c.learn()
