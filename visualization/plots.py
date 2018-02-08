import matplotlib
matplotlib.use('GTK3Agg') 
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(train_loss, val_loss, loss_name, path):
    fig = plt.subplot()
    fig.plot(train_loss, label='train')
    fig.plot(val_loss, label='val')
    fig.set(xlabel='epoch', ylabel='loss', title=loss_name)
    fig.set_ylim(0)    
    fig.legend()
    fig.grid()
    fig.figure.savefig(path + '{}.png'.format(loss_name))
    plt.close()

def save_plots(losses, epoch_num, path):
    plot_loss(losses[:epoch_num, 0], losses[:epoch_num, 5], 'loss_rpn_cls', path)
    plot_loss(losses[:epoch_num, 1], losses[:epoch_num, 6], 'loss_rpn_regr', path)
    plot_loss(losses[:epoch_num, 2], losses[:epoch_num, 7], 'loss_class_cls', path)
    plot_loss(losses[:epoch_num, 3], losses[:epoch_num, 8], 'loss_class_regr', path)
    plot_loss(losses[:epoch_num, 4], losses[:epoch_num, 9], 'class_acc', path)
    plot_loss(np.sum(losses[:epoch_num,:4],axis=1), np.sum(losses[:epoch_num,5:9],axis=1), 'total_loss', path)