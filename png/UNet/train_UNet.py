# Imports
import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from dependencies.DataSet import DataSet
from dependencies.PatchExtractor import PatchExtractor
from dependencies.BatchCreator import BatchCreator
from dependencies.data_loading import *


# Set paths to the data
path = os.getcwd()
parent = os.path.dirname(os.path.dirname(os.path.dirname(path)))

X_DIR = os.path.join(parent, r'data_sample\wsirois\roi-level-annotations\tissue-bcss\images')
y_DIR = os.path.join(parent, r'data_sample\wsirois\roi-level-annotations\tissue-bcss\masks')

rois_files = get_file_list(X_DIR, ext = '.png')
rois_lbls = get_file_list(y_DIR, ext = '.png')

train_rois_i = [load_img(f) for f in rois_files]
train_msks_i = [np.ones(roi.shape) for roi in train_rois_i]
train_rois = [reshape(f) for f in train_rois_i]
train_lbls = [np.squeeze(reshape(np.expand_dims(load_img(f), axis = 2))) for f in rois_lbls]
train_msks = [reshape(f)[:, :, 0] for f in train_msks_i]

# Define the number of validation images
n_validation_imgs = int(np.floor(0.2 * len(train_rois)))

# use the first images as validation
validation_data = DataSet(train_rois[:n_validation_imgs], train_msks[:n_validation_imgs], train_lbls[:n_validation_imgs])

# the rest as training
train_data = DataSet(train_rois[n_validation_imgs:], train_msks[n_validation_imgs:], train_lbls[n_validation_imgs:])

patch_size = (256, 256) # Set the size of the patches as a tuple (height, width) 
batch_size = 128        # pick a reasonable batch-size (e.g. power-of-two in the range 32, 64, 128, 256)
steps_per_epoch = 4            # how many steps per epoch?
epochs = 64                     # how many epochs? - Running more epochs simply took too long.
stride = 4                     # what is the downscaling factor of your network? stride = 3? originally 4


patch_extractor = PatchExtractor(patch_size, True)

batch_creator = BatchCreator(patch_extractor, train_data, patch_extractor.patch_size)

generator = batch_creator.get_generator(batch_size)
logger_1 = Logger(validation_data, patch_size, stride=stride)