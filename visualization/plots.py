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
    
def save_plots_from_history(rpn_hist, cls_hist, path):
    losses = np.zeros((len(rpn_hist), 14))
    epoch_num = len(rpn_hist)
    
            
    for epoch_num in range(len(rpn_hist)):
        losses[epoch_num, 0] = rpn_hist[epoch_num]['rpn_out_class_loss']
        losses[epoch_num, 1] = rpn_hist[epoch_num]['rpn_out_regress_loss']
        losses[epoch_num, 5] = rpn_hist[epoch_num]['val_rpn_out_class_loss']
        losses[epoch_num, 6] = rpn_hist[epoch_num]['val_rpn_out_regress_loss']
        losses[epoch_num, 10] = rpn_hist[epoch_num]['loss']
        losses[epoch_num, 11] = rpn_hist[epoch_num]['val_loss']
        
    for epoch_num in range(len(cls_hist)):
        losses[epoch_num, 2] = cls_hist[epoch_num]['dense_class_21_loss']
        losses[epoch_num, 3] = cls_hist[epoch_num]['dense_regress_21_loss']
        losses[epoch_num, 4] = cls_hist[epoch_num]['dense_class_21_acc']
        losses[epoch_num, 7] = cls_hist[epoch_num]['val_dense_class_21_loss']
        losses[epoch_num, 8] = cls_hist[epoch_num]['val_dense_regress_21_loss']
        losses[epoch_num, 9] = cls_hist[epoch_num]['val_dense_class_21_acc']
        losses[epoch_num, 12] = cls_hist[epoch_num]['loss']
        losses[epoch_num, 13] = cls_hist[epoch_num]['val_loss']
        
        
    plot_loss(losses[:epoch_num, 0], losses[:epoch_num, 5], 'loss_rpn_cls', path)
    plot_loss(losses[:epoch_num, 1], losses[:epoch_num, 6], 'loss_rpn_regr', path)
    plot_loss(losses[:epoch_num, 2], losses[:epoch_num, 7], 'loss_class_cls', path)
    plot_loss(losses[:epoch_num, 3], losses[:epoch_num, 8], 'loss_class_regr', path)
    plot_loss(losses[:epoch_num, 4], losses[:epoch_num, 9], 'class_acc', path)
    plot_loss(losses[:epoch_num, 10], losses[:epoch_num, 11], 'rpn_loss', path)
    plot_loss(losses[:epoch_num, 12], losses[:epoch_num, 13], 'detektor_loss', path)
    plot_loss(losses[:epoch_num, 10]+losses[:epoch_num, 12], losses[:epoch_num, 11]+losses[:epoch_num, 13], 'total_loss', path)