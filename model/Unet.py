from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, Cropping2D, Reshape, BatchNormalization, Dense
import os
import sys

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def unet_block(inputs, n_filters, batchnorm=False, dropout=False):
    
    c1 = Conv2D(n_filters, 3, activation='relu', padding='same')(inputs)
    if batchnorm: 
      c1 = BatchNormalization()(c1)
    if dropout:
      c1 = Dropout(0.3)(c1)
    c2 = Conv2D(n_filters, 3, activation='relu', padding='same')(c1)
    if batchnorm: 
      c2 = BatchNormalization()(c2)
    if dropout:
      c2 = Dropout(0.3)(c2)
    
    return c2

def build_unet(initial_filters=16, n_classes=2, batchnorm=False, dropout=False, printmodel=False):

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