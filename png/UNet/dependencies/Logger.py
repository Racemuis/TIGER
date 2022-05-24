import tensorflow.keras.callbacks
import numpy as np
import matplotlib.pyplot as plt

from dependencies.BatchCreator import pad

def downscale(images, stride):
    # Downscale if the network does pooling
    return np.array(images)[:, ::stride, ::stride]

def calculate_dice(x, y, class_x, class_y):
    '''returns the dice similarity score between two arrays, given the input classes'''
    #return 2 * np.count_nonzero(x & y) / (np.count_nonzero(x) + np.count_nonzero(y))
    tp = np.count_nonzero(x[x == y][x[x == y] == class_x]) 
    fp = np.count_nonzero(x[x != y][x[x != y] == class_x])
    fn = np.count_nonzero(x[x != y][x[x != y] != class_x])
    denom = (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1
    return 2 * tp/denom
    

def pad_ensure_division(h, w, division):

    def compute_pad(s, d):
        if s % d != 0:
            p = 0
            while True:
                if (s + p) % d == 0:
                    return p
                p += 1
        return 0

    py = compute_pad(h, division)
    px = compute_pad(w, division)
    padding = (py//2, py-py//2), (px//2, px-px//2)
    return padding
    
class UNetLogger(tensorflow.keras.callbacks.Callback):

    def __init__(self, validation_data):
        self.val_imgs = np.asarray(validation_data.imgs) / 255.
        self.val_lbls = np.asarray(validation_data.lbls) > 0
        self.val_msks = np.asarray(validation_data.msks) > 0

        self.losses = []
        self.dices = []
        self.best_dice = -1
        self.best_model = None

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, batch, logs={}):
        dice = self.validate()
        print(f" Dices: {dice}")
        self.dices.append([len(self.losses), dice])
        if np.mean(dice) > self.best_dice:
            print('updating the best model')
            self.best_dice = np.mean(dice)
            self.best_model = self.model#.get_weights()
        # self.plot()

    def validate(self):
        # need to pad image such that the size can be divisable by 8
        h, w = self.val_imgs.shape[1:3]
        (py0, py1), (px0, px1) = pad_ensure_division(h, w, 8)
        padding = ((0, 0), (py0, py1), (px0, px1), (0, 0))
        pad_val_imgs = np.pad(self.val_imgs, pad_width=padding, mode='constant')

        # run unet model
        predicted_lbls = np.argmax(self.model.predict(pad_val_imgs, batch_size=1), axis=-1)
        
        # crop it back because we pad it before
        b, h, w = predicted_lbls.shape
        predicted_lbls = predicted_lbls[:, py0:h-py1, px0:w-px1]

        x = self.val_lbls[self.val_msks]
        y = predicted_lbls[self.val_msks]
        
        return calculate_dice(x, y, 0, 2), calculate_dice(x, y, 1, 2)

    def plot(self):
        N = len(self.losses)
        train_loss_plt, = plt.plot(range(0, N), self.losses)
        dice_plt, = plt.plot(*np.array(self.dices).T)
        plt.legend((train_loss_plt, dice_plt),
                   ('training loss', 'validation dice'))
        plt.show()