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

train_data.show_image(0)

patch_size = (256, 256) # Set the size of the patches as a tuple (height, width) 

img_index = 0 # choose an image to extract the patch from
location = (1000, 1000) # define the location of the patch (y, x) - coordinate

patch_extractor = PatchExtractor(patch_size, True)

batch_creator = BatchCreator(patch_extractor, train_data, patch_extractor.patch_size)

# create a batch
x, y = batch_creator.create_batch(28)
# visualize it
f, axes = plt.subplots(4, 7)
i = 0
for ax_row in axes:
    for ax in ax_row:
        ax.imshow(x[i])
        ax.set_title('class: {}'.format(np.argmax(y[i, 0, 0])))
        ax.scatter(*[p/2 for p in patch_extractor.patch_size], alpha=0.5)
        i += 1
plt.show()