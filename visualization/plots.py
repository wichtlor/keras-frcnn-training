import matplotlib.pyplot as plt
import numpy as np
    
def plot_loss(loss_array, loss_name, path):
    fig, ax = plt.subplots()
    ax.plot(loss_array)
    ax.set(xlabel='epoch', ylabel='loss', title=loss_name)
    ax.grid()
    fig.savefig(path + '{}.png'.format(loss_name))
    plt.close()

def save_plots(losses, epoch_num, path):
    plot_loss(losses[:epoch_num, 0], 'train_loss_rpn_cls', path)
    plot_loss(losses[:epoch_num, 1], 'train_loss_rpn_regr', path)
    plot_loss(losses[:epoch_num, 2], 'train_loss_class_cls', path)
    plot_loss(losses[:epoch_num, 3], 'train_loss_class_regr', path)
    plot_loss(losses[:epoch_num, 4], 'train_class_acc', path)
    plot_loss(losses[:epoch_num, 5], 'val_loss_rpn_cls', path)
    plot_loss(losses[:epoch_num, 6], 'val_loss_rpn_regr', path)
    plot_loss(losses[:epoch_num, 7], 'val_loss_class_cls', path)
    plot_loss(losses[:epoch_num, 8], 'val_loss_class_regr', path)
    plot_loss(losses[:epoch_num, 9], 'val_class_acc', path)
    
    plot_loss(np.sum(losses[:epoch_num,:4],axis=1), 'total_train_loss', path)
    plot_loss(np.sum(losses[:epoch_num,5:9],axis=1), 'total_val_loss', path)