import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from utils.train_test_split import clean_train_test_split
from utils.DataSet import DataSet
from utils.PatchExtractor import PatchExtractor
from utils.BatchCreator import UNetBatchCreator
from utils.i_o import process_folder

from model.Unet import build_unet

# Disable the GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.optimizers import Adam
import tensorflow.keras.callbacks

# Specify data paths
X_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsibulk\images'
y_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsibulk\annotations-tumor-bulk\masks'
MSKS_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsibulk\tissue-masks'

# Create a train-test split
X_train_files, X_test_files, y_train_files, y_test_files, msks_train_files, msks_test_files = clean_train_test_split(X_directory=X_DIR, y_directory= y_DIR, test_size=0.4)

# Read the data
X_train = process_folder(path = X_DIR, targets=X_train_files)
y_train = process_folder(path=y_DIR, targets=y_train_files)
msks_train = process_folder(path = MSKS_DIR, targets = msks_train_files)

train_set = DataSet(X_train, msks_train, y_train)

X_test = process_folder(path= X_DIR, targets = X_test_files)
y_test = process_folder(path=y_DIR, targets=y_test_files)
msks_test = process_folder(path = MSKS_DIR, targets = msks_test_files)

test_set = DataSet(X_test, msks_test, y_test)

patch_size = (101, 101)
patch_extractor = PatchExtractor(patch_size, horizontal_flipping = True)
batch_creator = UNetBatchCreator(patch_extractor, train_set, patch_size)


def calculate_dice(x, y):
    '''returns the dice similarity score between two boolean arrays'''
    return 2 * np.count_nonzero(x & y) / (np.count_nonzero(x) + np.count_nonzero(y))

def pad_ensure_division(h, w, division):
    """
    Calculate padding that is divisible by a certain number

    Args:
        h (int): the height of the image
        w (int): the width of the image
        division (int): the required divisor
    
    Returns:
        padding: a set of tuples that can be used to pad the image
    """
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

def process_unet(model, imgs):
    outputs = []
    for img in imgs:
    # pad image if the size is not divisable by total downsampling rate in your U-Net 
        h, w, _ = img.shape
        (py0, py1), (px0, px1) = pad_ensure_division(h, w, 8)
        padding = ((py0, py1), (px0, px1), (0, 0))
        pad_img = np.pad(img, pad_width=padding, mode='constant')
    
    # run unet model
        output = model.predict(np.array([pad_img]), batch_size=1)[:, :, :, 1]
    
    # don't forget to crop it back because you pad it before.
        h, w, _ = img.shape
        outputs.append(output[:, py0:h+py1, px0:w+px0])

    return outputs

def check_results_unet(imgs, lbls, msks, output, threshold=0.5, plot = False):

    dices = []
    for i, (img, lbl, msk, raw_output) in enumerate(zip(imgs, lbls, msks, output)):
        final_output = np.where(raw_output.squeeze(0) > threshold, 1, 0)
        
        dice = calculate_dice(final_output, lbl.squeeze())
        dices.append(dice)
        print('image:', i, 'dice', dice)
        
        if(plot):
            # plot the results
            f, axes = plt.subplots(1, 4)
            for ax, im, t in zip(axes, 
                                    (img, raw_output.squeeze(0), final_output.astype(float), lbl.squeeze(-1)), 
                                    ('RGB image', 'Soft prediction', 'Thresholded', 'Ground truth')):
                ax.imshow(im, cmap='gray')
                ax.set_title(t)
            plt.show()
        
    print('mean dice', np.mean(dices))

class UNetLogger(tensorflow.keras.callbacks.Callback):

    def __init__(self, validation_data):
        self.val_imgs = np.asarray(validation_data.imgs) / 255.

        val_lbls = []
        val_msks = []

        for lbl, msk in zip(validation_data.lbls, validation_data.msks):
            val_lbls.append(lbl > 0)
            val_msks.append(msk > 0)
        
        self.val_lbls = val_lbls
        self.val_msks = val_msks

        self.losses = []
        self.dices = []
        self.best_dice = -1
        self.best_model = None

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, batch, logs={}):
        dice = self.validate()
        self.dices.append([len(self.losses), dice])
        if dice > self.best_dice:
            print('updating the best model')
            self.best_dice = dice
            self.best_model = self.model.get_weights()

    def validate(self):
        # run unet model
        predicted_lbls = []
        for img in self.val_imgs:
            h, w, _ = img.shape

            # need to pad image such that the size can be divisable by 8
            (py0, py1), (px0, px1) = pad_ensure_division(h, w, 8)
            padding = ((py0, py1), (px0, px1), (0, 0))
            padded = np.pad(img, pad_width=padding, mode='constant')

            prediction = np.argmax(self.model.predict(np.array([padded]), batch_size=1)[0], axis=-1)
            
            # crop it back because we pad it before
            predicted_lbls.append(prediction[py0:h+py1, px0:w+px1])

        x = []
        y = []
        for lbl, msk, preds in zip(self.val_lbls, self.val_msks, predicted_lbls):
            x.extend(lbl[msk].flatten())
            y.extend(preds[msk.squeeze(2)].flatten())
        x = np.array(x).flatten();y = np.array(y).flatten()
        return calculate_dice(x, y)

    def plot(self):
        N = len(self.losses)
        train_loss_plt, = plt.plot(range(0, N), self.losses)
        dice_plt, = plt.plot(*np.array(self.dices).T)
        plt.legend((train_loss_plt, dice_plt),
                   ('training loss', 'validation dice'))
        plt.show()

# Train UNet
learning_rate = 5e-4     
patch_size = (32,32)
batch_size = 16
steps_per_epoch = 50
epochs = 10

unet = build_unet(initial_filters=16, n_classes=2, batchnorm=True, dropout=True, printmodel=False)

optimizer = Adam(learning_rate)
unet.compile(optimizer, loss='categorical_crossentropy')

patch_extractor = PatchExtractor(patch_size, horizontal_flipping=True)
batch_creator = UNetBatchCreator(patch_extractor, train_set, patch_size)
patch_generator = batch_creator.get_generator(batch_size)
logger_4 = UNetLogger(test_set)

unet.fit(patch_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs, callbacks=[logger_4])

overwrite = True
if overwrite or not Path('./unet').exists():
    unet.set_weights(logger_4.best_model)
    unet.save('./unet')

imgs = [img/255. for img in test_set.imgs] 
lbls = []
msks = []

for lbl, msk in zip(test_set.lbls, test_set.msks):
    lbls.append(lbl > 0)
    msks.append(msk > 0)

output = process_unet(unet, imgs)
check_results_unet(test_set.imgs, lbls, msks, output, threshold=0.5)
