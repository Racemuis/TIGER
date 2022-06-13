# Adapted from Intelligent systems in medical imaging - Assignment 5

import numpy as np

class PatchExtractor:

    def __init__(self, patch_size, horizontal_flipping=True):
        self.patch_size = patch_size
        self.horizontal_flipping = horizontal_flipping
        
        
    def get_patch(self, image, mask, location):
        ''' 
        image: a numpy array representing the input image,
        mask: a numpy array representing the corresponding segmentation annotation
        location: a tuple with an y and x coordinate
        
        return a patch from the image at `location`, representing the center of the patch, and the corresponding label patch
        if self.horizontal_flipping = True, there is a 50% chance the patch is horizontally flipped  
        we will not rotate it or perform other augmentations for now to speed up the training process
        '''
        y, x = location
        py, px = self.patch_size
        # - patch should be a numpy array of size <h, w>
        # - the patch should be normalized (intensity values between 0-1)
    
        img_patch = image[y-py//2:y+(py-py//2), x-px//2:x+(px-px//2), :]
        label_patch = mask[y-py//2:y+(py-py//2), x-px//2:x+(px-px//2)]
        #img_patch = image[y:y+py, x:x+px, :]
        #label_patch = mask[y:y+py, x:x+px]
        img_patch = (img_patch - np.min(img_patch))/max(1,(np.max(img_patch)-np.min(img_patch)))
        label_patch = (label_patch - np.min(label_patch))/max(1,(np.max(label_patch)-np.min(label_patch)))
        
        # - if self.flipping = True, there should be a 50% chance to apply a horizontal flip to the patch  
        if self.horizontal_flipping:
            do_flipping = np.random.rand() >= 0.5
            
            # - if do_flipping == True, flip the patch horizontally
            if do_flipping:
                img_patch_flipped = np.flip(img_patch, 1)
                label_patch_flipped = np.flip(label_patch, 1)

                img_patch = img_patch_flipped
                label_patch = label_patch_flipped
        assert img_patch.ndim == label_patch.ndim
        return img_patch, label_patch
