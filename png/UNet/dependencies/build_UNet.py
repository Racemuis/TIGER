from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt

from dependencies.Logger import pad_ensure_division
def unet_block(inputs, n_filters, batchnorm=False, dropout=False):
    
    cl = Conv2D(n_filters, 3, activation='relu', padding='same')(inputs)
    if batchnorm:
        cl = BatchNormalization()(cl)
    if dropout:
        cl = Dropout(0.3)(cl)
    cl = Conv2D(n_filters, 3, activation='relu', padding='same')(inputs)
    if batchnorm:
        cl = BatchNormalization()(cl)
    if dropout:
        cl = Dropout(0.3)(cl)
    
    return cl

def build_unet(initial_filters=16, n_classes=3, batchnorm=False, dropout=False, printmodel=False):

    # build U-Net again using unet_block function
    inputs = Input(shape=(None, None, 3)) #adjust

    # CONTRACTION PART

    # First conv pool
    c1 = unet_block(inputs, initial_filters, batchnorm, dropout)
    p1 = MaxPooling2D()(c1)

    # Second conv pool
    c2 = unet_block(p1, 2*initial_filters, batchnorm, dropout)
    p2 = MaxPooling2D()(c2)

    # Third conv pool
    c3 = unet_block(p2, 4*initial_filters, batchnorm, dropout)
    p3 = MaxPooling2D()(c3)

    # Fourth conv
    c4 = unet_block(p3, 8*initial_filters, batchnorm, dropout)

    # EXPANSION PART

    # First up-conv
    u2 = UpSampling2D()(c4)
    m2 = concatenate([c3, u2])
    cm2 = unet_block(m2, 4*initial_filters, batchnorm, dropout)

    # Second up-conv
    u3 = UpSampling2D()(cm2)
    m3 = concatenate([c2, u3])
    cm3 = unet_block(m3, 2*initial_filters, batchnorm, dropout)

    # Third up-conv
    u4 = UpSampling2D()(cm3)
    m4 = concatenate([c1, u4])
    cm4 = unet_block(m4, initial_filters, batchnorm, dropout)

    # Output
    predictions = Conv2D(n_classes, 1, activation='softmax')(cm4)

    model = Model(inputs, predictions)
    
    if printmodel:
        print(model.summary())
    
    return model

def calculate_dice(x, y):
    '''returns the dice similarity score, between two boolean arrays'''
    return 2 * np.count_nonzero(x & y) / (np.count_nonzero(x) + np.count_nonzero(y))

def check_results_unet(imgs, lbls, msks, output, threshold=0.5):

    dices = []
    for i, (img, lbl, _, raw_output) in enumerate(zip(imgs, lbls, msks, output)):
        final_output = raw_output*(255/3)
        
#         print('Output_mask: {}\nLbl: {}'.format(final_output[3].shape, lbl.shape))
        dice = calculate_dice(raw_output.flatten(), lbl.flatten())
        dices.append(dice)
        print('image:', i, 'dice', dice)
        
        # plot the results
        f, axes = plt.subplots(1, 4)
        for ax, im, t in zip(axes, 
                             (img, raw_output, final_output, lbl), 
                             ('RGB image', 'Soft prediction', 'Thresholded', 'Ground truth')):
            ax.imshow(im, cmap='gray')
            ax.set_title(t)
        plt.show()
        
    print('mean dice', np.mean(dices))

def process_unet(model, imgs):
    # pad image if the size is not divisable by total downsampling rate in your U-Net 
    _, h, w, _ = imgs.shape
    (py0, py1), (px0, px1) = pad_ensure_division(h, w, 8)
    padding = ((0, 0), (py0, py1), (px0, px1), (0, 0))
    pad_imgs = np.pad(imgs, pad_width=padding, mode='constant')
    
    # run unet model
    output = np.argmax(model.predict(pad_imgs, batch_size=1), axis = -1)+1
    
    # don't forget to crop it back because you pad it before.
    _, h, w, _ = imgs.shape
    output = output[:, py0:h-py1, px0:w+1]#-px1]

    return output