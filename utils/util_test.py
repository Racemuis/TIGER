import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from train_test_split import clean_train_test_split
from DataSet import DataSet
from PatchExtractor import PatchExtractor
from BatchCreator import UNetBatchCreator
from i_o import process_folder

# You possibly need to uncomment this once
# import sys
# sys.path.append(r'Your\path\to\TIGER')
from model.Unet import build_unet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.optimizers import Adam
import tensorflow.keras.callbacks


X_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsibulk\images'
y_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsibulk\annotations-tumor-bulk\masks'
MSKS_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsibulk\tissue-masks'
X_train_files, X_test_files, y_train_files, y_test_files, msks_train_files, msks_test_files = clean_train_test_split(X_directory=X_DIR, y_directory= y_DIR, test_size=0.4)

print(X_train_files, X_test_files)

X_train = process_folder(path = X_DIR, targets=X_train_files)
y_train = process_folder(path=y_DIR, targets=y_train_files)
msks_train = process_folder(path = MSKS_DIR, targets = msks_train_files)

train_set = DataSet(X_train, msks_train, y_train)

X_test = process_folder(path= X_DIR, targets = X_test_files)
y_test = process_folder(path=y_DIR, targets=y_test_files)
msks_test = process_folder(path = MSKS_DIR, targets = msks_test_files)

test_set = DataSet(X_test, msks_test, y_test)

# test_set.show_image(0) # change this parameter to try a few images
# train_set.show_image(0) # change this parameter to try a few images

patch_size = (101, 101) # Set the size of the patches as a tuple (height, width) 
img_index = 0 # choose an image to extract the patch from
location = (100, 100) # define the location of the patch (y, x) - coordinate

patch_extractor = PatchExtractor(patch_size, True)
image_patch, label_patch = patch_extractor.get_patch(train_set.imgs[img_index], train_set.lbls[img_index], location)
batch_creator = UNetBatchCreator(patch_extractor, train_set, patch_size)

"""
X, y = batch_creator.create_batch(5)

f, axes = plt.subplots(4, 5)
i = 0
for ax_row in axes:
    for ax in ax_row:
        ax.imshow(X[i])
        ax.set_title('class: {}'.format(np.argmax(y[i, 0, 0])))
        ax.scatter(*[p/2 for p in patch_extractor.patch_size], alpha=0.5)
        i += 1
plt.show()
"""

# Quick and dirty from the notebook: 

def calculate_dice(x, y):
    '''returns the dice similarity score, between two boolean arrays'''
    #print(x)
    #print(y)
    return 2 * np.count_nonzero(x & y) / (np.count_nonzero(x) + np.count_nonzero(y))

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

        val_lbls = []
        val_msks = []

        for lbl, msk in zip(validation_data.lbls, validation_data.msks):
            val_lbls.append(lbl > 0)
            val_msks.append(msk > 0)
        
        self.val_lbls = val_lbls
        self.val_msks = val_msks
        
        #self.val_lbls = np.asarray(validation_data.lbls) > 0
        #self.val_msks = np.asarray(validation_data.msks) > 0

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
        self.plot()

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

learning_rate = 5e-4      #Set a learning rate
patch_size = (32,32)
batch_size = 16
steps_per_epoch = 50
epochs = 10


unet_2 = build_unet(initial_filters=16, n_classes=2, batchnorm=True, dropout=True, printmodel=False)

optimizer = Adam(learning_rate)
unet_2.compile(optimizer, loss='categorical_crossentropy')

patch_extractor = PatchExtractor(patch_size, horizontal_flipping=True)
batch_creator = UNetBatchCreator(patch_extractor, train_set, patch_size)
patch_generator = batch_creator.get_generator(batch_size)
logger_4 = UNetLogger(test_set)


unet_2.fit(patch_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs, callbacks=[logger_4])


overwrite = True
if overwrite or not Path('./unet_2').exists():
    unet_2.set_weights(logger_4.best_model)
    unet_2.save('./unet_2')

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

def check_results_unet(imgs, lbls, msks, output, threshold=0.5):

    dices = []
    for i, (img, lbl, msk, raw_output) in enumerate(zip(imgs, lbls, msks, output)):
        final_output = np.where(raw_output.squeeze(0) > threshold, 1, 0)
        print(raw_output.squeeze(0))
        print(final_output)
        
    #         print('Output_mask: {}\nLbl: {}'.format(final_output[3].shape, lbl.shape))
        print(f"Final output {final_output.shape}, labels {lbl.shape}")
        dice = calculate_dice(final_output, lbl.squeeze())
        dices.append(dice)
        print('image:', i, 'dice', dice)
        
        # plot the results
        f, axes = plt.subplots(1, 4)
        for ax, im, t in zip(axes, 
                                (img, raw_output.squeeze(0), final_output.astype(float), lbl.squeeze(-1)), 
                                ('RGB image', 'Soft prediction', 'Thresholded', 'Ground truth')):
            ax.imshow(im, cmap='gray')
            ax.set_title(t)
        plt.show()
        
    print('mean dice', np.mean(dices))
imgs = np.asarray(test_set.imgs) / 255.

lbls = []
msks = []

for lbl, msk in zip(test_set.lbls, test_set.msks):
    lbls.append(lbl > 0)
    msks.append(msk > 0)

output = process_unet(unet_2, imgs)
for out in output:
    print(out.shape)
check_results_unet(test_set.imgs, lbls, msks, output, threshold=0.5)
