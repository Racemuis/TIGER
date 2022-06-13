# Imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np

from dependencies.DataSet import DataSet
from dependencies.PatchExtractor import PatchExtractor
from dependencies.BatchCreator import UNetBatchCreator
from dependencies.data_loading import get_file_list, load_img, reshape
from dependencies.Logger import UNetLogger
from dependencies.build_UNet import build_unet, check_results_unet, process_unet
from tensorflow.keras.optimizers import Adam

# Set paths to the data (change this if nessecary)
X_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsirois\roi-level-annotations\tissue-cells\images'
y_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsirois\roi-level-annotations\tissue-cells\masks'

rois_files = get_file_list(X_DIR, ext = '.png')
rois_lbls = get_file_list(y_DIR, ext = '.png')

train_rois = [load_img(f) for f in rois_files]
train_lbls = [load_img(f) for f in rois_lbls]
train_msks = [np.zeros(f.shape) for f in train_rois]

# Define the number of validation images
n_validation_imgs = int(np.floor(0.2 * len(train_rois)))

# Create datasets
validation_data = DataSet(train_rois[:n_validation_imgs], train_msks[:n_validation_imgs], train_lbls[:n_validation_imgs])
train_data = DataSet(train_rois[n_validation_imgs:], train_msks[n_validation_imgs:], train_lbls[n_validation_imgs:])

# Visualize data
train_data.show_image(0)