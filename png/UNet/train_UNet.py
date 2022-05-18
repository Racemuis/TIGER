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

# Set paths to the data
path = os.getcwd()
parent = os.path.dirname(os.path.dirname(path))

# X_DIR = os.path.join(parent, r'data_sample\wsirois\roi-level-annotations\tissue-bcss\images')
# y_DIR = os.path.join(parent, r'data_sample\wsirois\roi-level-annotations\tissue-bcss\masks')

X_DIR = r'C:\Users\celen\Documents\Radboud year 1\Intelligent Systems in Medical Imaging\TIGER\data_sample\wsirois\roi-level-annotations\tissue-bcss\images'
y_DIR = r'C:\Users\celen\Documents\Radboud year 1\Intelligent Systems in Medical Imaging\TIGER\data_sample\wsirois\roi-level-annotations\tissue-bcss\masks'

rois_files = get_file_list(X_DIR, ext = '.png')
rois_lbls = get_file_list(y_DIR, ext = '.png')

train_rois_i = [load_img(f) for f in rois_files]

# Saving the shapes of the input roi's
train_size_rois = []
for img in train_rois_i: 
    train_size_rois.append(img.shape[:2])

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

learning_rate = 5e-4     
patch_size = (128, 128)
batch_size = 32
steps_per_epoch = 2
epochs = 3

patch_extractor = PatchExtractor(patch_size, horizontal_flipping=True)
batch_creator = UNetBatchCreator(patch_extractor, train_data, patch_size)
patch_generator = batch_creator.get_generator(batch_size)
logger = UNetLogger(validation_data)

unet = build_unet(initial_filters=16, n_classes=2, batchnorm=True, dropout=True, printmodel=True)

optimizer = Adam(learning_rate)
unet.compile(optimizer, loss='categorical_crossentropy')

unet.fit(patch_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[logger])

logger.best_model.save("UNet")


output = process_unet(unet, np.array(validation_data.imgs))

for i, elem in enumerate(train_size_rois): 
    x, y = elem[0], elem[1]
    output[i] = output[i][:x][:y]

print(output)
check_results_unet(validation_data.imgs, validation_data.lbls, reshape_valid, output=np.array(output), threshold= 0.1)