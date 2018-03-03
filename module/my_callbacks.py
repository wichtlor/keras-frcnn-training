from keras.callbacks import Callback
from keras import backend as K
import numpy as np

class LRReducer(Callback):
    def __init__(self, monitor='val_loss', factor=0.1, patience=10, epsilon=1e-4, min_lr=0):
                     
        super(LRReducer, self).__init__()
        
        self.monitor = monitor
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.wait = 0
        self.best = np.Inf
        self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        
        if  self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                old_lr = float(K.get_value(self.model.optimizer.lr))
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    K.set_value(self.model.optimizer.lr, new_lr)
                    print('ReduceLROnPlateau reducing learning rate to {}.'.format(new_lr))
                    self.wait = 0
            self.wait += 1