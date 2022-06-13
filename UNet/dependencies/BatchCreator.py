# Adapted from Intelligent systems in medical imaging - Assignment 5
import numpy as np
from dependencies.Samplers import BalancedSampler, Uniform_Sampler
from tensorflow.keras.utils import to_categorical


def pad(images, patch_size):
    '''
    images: list of images (numpy arrays)
    returns a padded version of the images, with a border of half the patch_size around each image
    '''
    half_py, half_px = [p//2 for p in patch_size]
    paddings = ((0, 0), (half_py, half_py), (half_px, half_px), (0, 0))
    return np.pad(np.array(images), pad_width=paddings, mode='constant', constant_values = 0) 

class BatchCreator:
    def __init__(self, patch_extractor, dataset, border_pad_size):
        self.patch_extractor = patch_extractor

        # Avoid extracting patches from the border that extend beyond the original image
        self.imgs = pad(dataset.imgs, border_pad_size)
        self.lbls = pad(np.expand_dims(dataset.lbls, 3), border_pad_size)
        self.msks = pad(np.expand_dims(dataset.msks, 3), border_pad_size)
        self.patch_extractor = patch_extractor
        self.patch_location_sampler = Uniform_Sampler(self.lbls, self.msks, border_pad_size)

    def create_batch(self, batch_size):
        '''
        returns a class-balanced array of patches (x) with corresponding labels (y) in one-hot structure
        '''
        x_data = np.zeros((batch_size, *self.patch_extractor.patch_size, 3))
        y_data = np.zeros((batch_size, 1, 1, 2))  # one-hot encoding
        locations = self.patch_location_sampler.generate_sample_locations(batch_size)
        
        for i, l in enumerate(locations):
            index, y, x = l
            print(y, x, self.imgs[index].shape)
            x_data[i], y_full = self.patch_extractor.get_patch(self.imgs[index], self.lbls[index], (y,x))
            label = y_full[y_full.shape[0]//2, y_full.shape[1]//2].squeeze()%254
            y_data[i] = np.array([1-label, label])
            
        return x_data, y_data
        
    def get_generator(self, batch_size):
        '''returns a generator that will yield batches infinitely'''
        while True:
            yield self.create_batch(batch_size)

class UNetBatchCreator(BatchCreator):

    def __init__(self, patch_extractor, dataset, border_pad_size):
        super(UNetBatchCreator, self).__init__(patch_extractor, dataset,
                                                    border_pad_size)

        self.patch_location_sampler = Uniform_Sampler(self.lbls, self.msks, border_pad_size)
    
    def create_batch(self, batch_size):
        '''
        returns a batch of image patches (x) with corresponding label patches (y) in one-hot structure
        '''
        x_data = np.zeros((batch_size, *self.patch_extractor.patch_size, 3))
        y_data = np.zeros((batch_size, *self.patch_extractor.patch_size, 3))  # one-hot encoding

        locations = self.patch_location_sampler.generate_sample_locations(batch_size)
        
        for i, l in enumerate(locations):
            index, y, x = l
            x_data[i], y_out = self.patch_extractor.get_patch(self.imgs[int(index)], self.lbls[int(index)], (int(y),int(x)))
            y_data[i] = to_categorical(y_out, num_classes = 3)
                    
        return x_data, y_data