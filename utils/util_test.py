import matplotlib.pyplot as plt
import numpy as np

from train_test_split import clean_train_test_split
from DataSet import DataSet
from PatchExtractor import PatchExtractor
from BatchCreator import UNetBatchCreator
from i_o import process_folder

X_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsibulk\images'
y_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsibulk\annotations-tumor-bulk\masks'
MSKS_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsibulk\tissue-masks'
X_train_files, X_test_files, y_train_files, y_test_files, msks_train_files, msks_test_files = clean_train_test_split(X_directory=X_DIR, y_directory= y_DIR, test_size=0.2)

print(X_train_files, X_test_files)

X_train = process_folder(path = X_DIR, targets=X_train_files)
y_train = process_folder(path=y_DIR, targets=y_train_files).astype(float)
msks_train = process_folder(path = MSKS_DIR, targets = msks_train_files).astype(float)

train_set = DataSet(X_train, y_train, msks_train)

X_test = process_folder(path= X_DIR, targets = X_test_files)
y_test = process_folder(path=y_DIR, targets=y_test_files).astype(float)
msks_test = process_folder(path = MSKS_DIR, targets = msks_test_files).astype(float)

test_set = DataSet(X_test, y_test, msks_test)

print(test_set.lbls.shape, test_set.lbls.squeeze().shape, test_set.imgs.shape)
# test_set.show_image(0) # change this parameter to try a few images
# train_set.show_image(0) # change this parameter to try a few images

patch_size = (101, 101) # Set the size of the patches as a tuple (height, width) 
img_index = 0 # choose an image to extract the patch from
location = (100, 100) # define the location of the patch (y, x) - coordinate

patch_extractor = PatchExtractor(patch_size, True)
image_patch, label_patch = patch_extractor.get_patch(train_set.imgs[img_index], train_set.lbls[img_index], location)
batch_creator = UNetBatchCreator(patch_extractor, train_set, patch_size)

X, y = batch_creator.create_batch(10)
f, axes = plt.subplots(2, 5)
i = 0
for ax_row in axes:
    for ax in ax_row:
        ax.imshow(X[i])
        ax.set_title('class: {}'.format(np.argmax(y[i, 0, 0])))
        ax.scatter(*[p/2 for p in patch_extractor.patch_size], alpha=0.5)
        i += 1
plt.show()


