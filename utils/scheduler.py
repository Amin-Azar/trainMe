
class ReduceLROnPlateau():
    """
    Reduce the learning rate if the train or validation loss plateaus
    """

    def __init__(self,
                 monitor='val_loss', 
                 factor=0.2, 
                 patience=10,
                 epsilon=0, 
                 cooldown=0, 
                 min_lr=0,
                 verbose=0):
        """
        Reduce the learning rate if the train or validation loss plateaus

        Arguments
        ---------
        monitor : string in {'loss', 'val_loss'}
            which metric to monitor
        factor : floar
            factor to decrease learning rate by
        patience : integer
            number of epochs to wait for loss improvement before reducing lr
        epsilon : float
            how much improvement must be made to reset patience
        cooldown : integer 
            number of epochs to cooldown after a lr reduction
        min_lr : float
            minimum value to ever let the learning rate decrease to
        verbose : integer
            whether to print reduction to console
        """
        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.wait = 0
        self.best_loss = 1e15
        self.optimizer = None
        self._reset()
        print("ReduceOnPlateau Scheduler Ready!")
        super(ReduceLROnPlateau, self).__init__()

    def _reset(self):
        """
        Reset the wait and cooldown counters
        """
        self.monitor_op = lambda a, b: (a - b) < -self.epsilon
        self.best_loss = 1e15
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, optimizer=None, logs=None):
        self.optimizer = optimizer
        self._reset()

    def on_epoch_end(self, epoch, monitor, logs=None):
        logs = logs or {}
        logs['lr'] = [p['lr'] for p in self.optimizer.param_groups]
        current_loss =  monitor[self.monitor]
        if current_loss is None:
            pass
        else:
            # if in cooldown phase
            if self.cooldown_counter > 0: 
                self.cooldown_counter -= 1
                self.wait = 0
            # if loss improved, grab new loss and reset wait counter
            if self.monitor_op(current_loss, self.best_loss):
                self.best_loss = current_loss
                self.wait = 0
                print('\nEpoch %05d: Acc Improved => keeping lr at %0.3f' %
                    (epoch, self.optimizer.param_groups[0]['lr']) )
            # loss didnt improve, and not in cooldown phase
            elif not (self.cooldown_counter > 0):
                if self.wait >= self.patience:
                    for p in self.optimizer.param_groups:
                        old_lr = p['lr']
                        if old_lr > self.min_lr + 1e-4:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            if self.verbose > 0:
                                print('\nEpoch %05d: reducing lr from %0.3f to %0.3f' % 
                                    (epoch, old_lr, new_lr))
                            p['lr'] = new_lr
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                else:
                    print('\nEpoch %05d: Patience! keeping lr at %0.3f' %
                        (epoch,self.optimizer.param_groups[0]['lr']))
                self.wait += 1

