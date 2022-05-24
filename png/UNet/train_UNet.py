# Imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import matplotlib.pyplot as plt

from dependencies.DataSet import DataSet
from dependencies.PatchExtractor import PatchExtractor
from dependencies.BatchCreator import UNetBatchCreator
from dependencies.data_loading import get_file_list, load_img, reshape, get_contents
from dependencies.Logger import UNetLogger
from dependencies.build_UNet import build_unet, check_results_unet, process_unet
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# Set paths to the data
#gets three levels up 
os.chdir(os.path.dirname(os.getcwd()))
os.chdir(os.path.dirname(os.getcwd()))
os.chdir(os.path.dirname(os.getcwd()))
path = os.getcwd()

X_DIR = os.path.join(path, r'project/data_sample\wsirois\roi-level-annotations\tissue-bcss\images')
y_DIR =  os.path.join(path,r'project/data_sample\wsirois\roi-level-annotations\tissue-bcss\masks')
#Chiara directory
#X_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsirois\roi-level-annotations\tissue-bcss\images'
#y_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsirois\roi-level-annotations\tissue-bcss\masks'

# Celena directory
# X_DIR = r'C:\Users\celen\Documents\Radboud year 1\Intelligent Systems in Medical Imaging\TIGER\data_sample\wsirois\roi-level-annotations\tissue-bcss\images'
# y_DIR = r'C:\Users\celen\Documents\Radboud year 1\Intelligent Systems in Medical Imaging\TIGER\data_sample\wsirois\roi-level-annotations\tissue-bcss\masks'

TIFF_READING_LEVEL = 5 # Available levels: 0-6
rois_files = get_file_list(X_DIR, ext = '.png')
rois_lbls = [os.path.join(y_DIR, f) for f in get_contents(X_DIR, ext = '.png')]#get_file_list(y_DIR, ext = '.png')

train_rois_i = [load_img(f, TIFF_READING_LEVEL) for f in tqdm(rois_files, desc= "Loading images")]
train_msks_i = [np.ones(roi.shape) for roi in tqdm(train_rois_i, desc = "Creating masks")]
train_rois = [reshape(f)/255 for f in tqdm(train_rois_i, desc = "Reshaping images")]
train_lbls = [np.squeeze(reshape(np.expand_dims(load_img(f, TIFF_READING_LEVEL), axis = 2))) for f in tqdm(rois_lbls, desc = "Loading labels")]

train_msks = [reshape(f)[:, :, 0].astype(float) for f in tqdm(train_msks_i, desc = "Reshaping masks")]

# Define the number of validation images
n_validation_imgs = int(np.floor(0.2 * len(train_rois)))

# use the first images as validation
validation_data = DataSet(train_rois[:n_validation_imgs], train_msks[:n_validation_imgs], train_lbls[:n_validation_imgs])

# Saving the shapes of the validation roi's
validation_size_roi = []
for img in train_msks_i[:n_validation_imgs]: 
    validation_size_roi.append(img.shape[:2])

# the rest as training
train_data = DataSet(train_rois[n_validation_imgs:], train_msks[n_validation_imgs:], train_lbls[n_validation_imgs:])

# for i, _ in enumerate(train_data.imgs):
#    train_data.show_image(i)

# Free some memory
del train_rois_i
del train_msks_i
del train_rois
del train_lbls
del train_msks 

learning_rate = 5e-4   
patch_size = (64, 64)
batch_size = 64
steps_per_epoch = 60
epochs = 5

patch_extractor = PatchExtractor(patch_size, horizontal_flipping=True)
batch_creator = UNetBatchCreator(patch_extractor, train_data, patch_size)
patch_generator = batch_creator.get_generator(batch_size)
logger = UNetLogger(validation_data)

unet = build_unet(initial_filters=16, n_classes=8, batchnorm=True, dropout=True, printmodel=True)

optimizer = Adam(learning_rate)
unet.compile(optimizer, loss='categorical_crossentropy')

unet.fit(patch_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[logger])

logger.best_model.save("Best_UNet")

unet.save("Final_UNet")

output = process_unet(unet, np.array(validation_data.imgs))

new_out = []
new_lbl = []
new_img = []
for i, elem in enumerate(validation_size_roi): 
    x, y = elem[0], elem[1]
    new_out.append(output[i][:x,:y])
    new_lbl.append(validation_data.lbls[i][:x,:y])
    new_img.append(validation_data.imgs[i][:x,:y])

check_results_unet(new_img, new_lbl, validation_data.msks, new_out, threshold= 0.1)
