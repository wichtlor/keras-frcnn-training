import matplotlib.pyplot as plt
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool 

def plot_loss(args):
    fig = plt.subplot()
    fig.plot(args[0], label='train')
    fig.plot(args[1], label='val')
    fig.set(xlabel='epoch', ylabel='loss', title=args[2])
    fig.set_ylim(0)    
    fig.legend()
    fig.grid()
    fig.figure.savefig(args[3] + '{}.png'.format(args[2]))
    plt.close()

def save_plots(losses, epoch_num, path):
    plot = [
        [losses[:epoch_num, 0], losses[:epoch_num, 5], 'loss_rpn_cls', path],
        [losses[:epoch_num, 1], losses[:epoch_num, 6], 'loss_rpn_regr', path],
        [losses[:epoch_num, 2], losses[:epoch_num, 7], 'loss_class_cls', path],
        [losses[:epoch_num, 3], losses[:epoch_num, 8], 'loss_class_regr', path],
        [losses[:epoch_num, 4], losses[:epoch_num, 9], 'class_acc', path],
        [np.sum(losses[:epoch_num,:4],axis=1), np.sum(losses[:epoch_num,5:9],axis=1), 'total_loss', path]
    ]    
    
    pool = ThreadPool(6)
    pool.map(plot_loss, plot)
    pool.close()
    pool.join()

def test():
    epoch_num = 1000
    path = './'
    losses = np.zeros((2000, 10))
    plot = [
        [losses[:epoch_num, 0], losses[:epoch_num, 5], 'loss_rpn_cls', path],
        [losses[:epoch_num, 1], losses[:epoch_num, 6], 'loss_rpn_regr', path],
        [losses[:epoch_num, 2], losses[:epoch_num, 7], 'loss_class_cls', path],
        [losses[:epoch_num, 3], losses[:epoch_num, 8], 'loss_class_regr', path],
        [losses[:epoch_num, 4], losses[:epoch_num, 9], 'class_acc', path],
        [np.sum(losses[:epoch_num,:4],axis=1), np.sum(losses[:epoch_num,5:9],axis=1), 'total_loss', path]
    ]
    
    pool = ThreadPool(6)
    pool.map(plot_loss, plot)
    pool.close()
    pool.join()