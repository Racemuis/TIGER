import tensorflow.keras.callbacks
import numpy as np
import matplotlib.pyplot as plt

from dependencies.BatchCreator import pad

def downscale(images, stride):
    # Downscale if the network does pooling
    return np.array(images)[:, ::stride, ::stride]

def calculate_dice(x, y):
    '''returns the dice similarity score, between two boolean arrays'''
    return 2 * np.count_nonzero(x & y) / (np.count_nonzero(x) + np.count_nonzero(y))
    
class Logger(tensorflow.keras.callbacks.Callback):

    def __init__(self, validation_data, patch_size, stride=1):
        self.val_imgs = pad(validation_data.imgs, patch_size) / 255.
        self.val_lbls = downscale(validation_data.lbls, stride) > 0
        self.val_msks = downscale(validation_data.msks, stride) > 0
         
        self.losses = []
        self.dices = []
        self.best_dice = 0
        self.best_model = None
    
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
    
    def on_epoch_end(self, batch, logs={}):
        dice = self.validate()
        self.dices.append([len(self.losses), dice])
        if dice > self.best_dice:
            self.best_dice = dice
            self.best_model = self.model.get_weights()
        self.plot()
           
    def validate(self):
        predicted_lbls = self.model.predict(self.val_imgs, batch_size=1)[:,:,:,1]>0.5
        x = self.val_lbls[self.val_msks]
        y = predicted_lbls[self.val_msks]
        return calculate_dice(x, y)
    
    def plot(self):
        clear_output()
        N = len(self.losses)
        train_loss_plt, = plt.plot(range(0, N), self.losses)
        dice_plt, = plt.plot(*np.array(self.dices).T)
        plt.legend((train_loss_plt, dice_plt), 
                   ('training loss', 'validation dice'))
        plt.show()