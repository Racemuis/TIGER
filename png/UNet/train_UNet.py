# Imports
import os
import numpy as np
from operator import itemgetter
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import CategoricalAccuracy

from dependencies.DataSet import DataSet
from dependencies.PatchExtractor import PatchExtractor
from dependencies.BatchCreator import UNetBatchCreator
from dependencies.data_loading import get_file_list, load_img, reshape, get_contents
from dependencies.Logger import UNetLogger, categorical_dice
from dependencies.build_UNet import build_unet, check_results_unet, process_unet

# Hyperparameters
# Set paths to the data
os.chdir(os.path.dirname(os.getcwd()))
os.chdir(os.path.dirname(os.getcwd()))
os.chdir(os.path.dirname(os.getcwd()))
path = os.getcwd()

X_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsirois\roi-level-annotations\tissue-bcss\images'
y_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsirois\roi-level-annotations\tissue-bcss\masks'

CLUSTER_MODE = False
TIFF_READING_LEVEL = 5 # Available levels: 0-6

# Load data
rois_files = get_file_list(X_DIR, ext = '.png')
rois_lbls = [os.path.join(y_DIR, f) for f in get_contents(X_DIR, ext = '.png')]

train_rois_i = [load_img(f, TIFF_READING_LEVEL) for f in tqdm(rois_files, desc= "Loading images")]
train_msks_i = [np.ones(roi.shape) for roi in tqdm(train_rois_i, desc = "Creating masks")]

# Get largest axis
max_x = max([img.shape[0] for img in train_rois_i])
max_y = max([img.shape[1] for img in train_rois_i])
dim = max(max_x, max_y)

# Disable GPU
if not CLUSTER_MODE:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    dim = 128

train_rois = [reshape(f, dim) for f in tqdm(train_rois_i, desc = "Reshaping images")]
train_lbls = [np.squeeze(reshape(np.expand_dims(load_img(f, TIFF_READING_LEVEL), axis = 2), dim)) for f in tqdm(rois_lbls, desc = "Loading labels")]
train_msks = [reshape(f, dim)[:, :, 0].astype(float) for f in tqdm(train_msks_i, desc = "Reshaping masks")]

# LABEL REMAPPING: 
#         old label -> new label
#             2 & 6 -> 2
#                   -> 1
# 3, 4, 5, 7 + mask -> 0
converted_lbls = []
for lbl in train_lbls:
    new_lbl = np.zeros(lbl.shape)
    new_lbl[lbl==2] = 2
    new_lbl[lbl==6] = 2
    new_lbl[lbl==1] = 1
    converted_lbls.append(new_lbl.astype(int))

# Define the number of validation images
n_validation_imgs = int(np.floor(0.2 * len(train_rois)))

# Saving the shapes of the validation roi's
validation_size_roi = []
for img in train_msks_i[:n_validation_imgs]: 
    validation_size_roi.append(img.shape[:2])

# Create datasets
validation_data = DataSet(train_rois[:n_validation_imgs], train_msks[:n_validation_imgs], converted_lbls[:n_validation_imgs])    
train_data = DataSet(train_rois[n_validation_imgs:], train_msks[n_validation_imgs:], converted_lbls[n_validation_imgs:])

# Free some memory
del train_rois_i
del train_msks_i
del train_rois
del train_lbls
del train_msks 

# Set hyperparameters for UNet
learning_rate = 5e-4   

if CLUSTER_MODE:
    patch_size = (256, 256)
    batch_size = 128
    steps_per_epoch = 60
    epochs = 40

else: 
    patch_size = (64, 64)
    batch_size = 64
    steps_per_epoch = 60
    epochs = 5

patch_extractor = PatchExtractor(patch_size, horizontal_flipping=True)
batch_creator = UNetBatchCreator(patch_extractor, train_data, patch_size)
patch_generator = batch_creator.get_generator(batch_size)
logger = UNetLogger(validation_data)

unet = build_unet(initial_filters=16, n_classes=3, batchnorm=True, dropout=True, printmodel=True)

optimizer = Adam(learning_rate)
unet.compile(optimizer, loss='categorical_crossentropy', metrics=[categorical_dice])


checkpoint_filepath ='Tiger/png/UNet/saved_model/model_Unet.h5'

model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_categorical_dice',
    mode='max',
    save_best_only=False)


unet.fit(patch_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs, validation_data = (np.array(validation_data.imgs), 
        np.array(to_categorical(validation_data.lbls, num_classes = 3))), 
        callbacks=[logger, model_checkpoint_callback])

unet2 = build_unet(initial_filters=16, n_classes=3, batchnorm=True, dropout=True, printmodel=True)
unet2.load_weights("./"+checkpoint_filepath)

# Show UNet output
output = process_unet(unet2, np.array(validation_data.imgs))

new_out = []
new_lbl = []
new_img = []
for i, elem in enumerate(validation_size_roi): 
    # Remove the padding
    x, y = elem[0], elem[1]
    new_out.append(output[i][:x,:y])
    new_lbl.append(validation_data.lbls[i][:x,:y])
    new_img.append(validation_data.imgs[i][:x,:y])

check_results_unet(new_img, new_lbl, validation_data.msks, new_out, threshold= 0.1)
