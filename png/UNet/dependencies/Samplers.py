import numpy as np
import random

class BalancedSampler:
# Adapted from Intelligent systems in medical imaging - Assignment 5

    def __init__(self, lbls, msks, border_pad_size):
        # pre calculate the positive and negative indices
        lbls = np.squeeze(lbls, 3)
        msks = np.squeeze(msks, 3)
        zero_idxs = np.asarray(np.where((msks !=0) & lbls == 0)).T.tolist()
        one_idxs = np.asarray(np.where((msks !=0) & lbls == 1)).T.tolist()

        # Remove samples that are too close to the padding:
        lower_l = border_pad_size[0]//2 
        upper_l = lbls.shape[1]-border_pad_size[0]//2
        lower_r = border_pad_size[1]//2 
        upper_r = lbls.shape[2]-border_pad_size[1]//2
        self.zero_idxs = [idx for idx in zero_idxs if idx[1] > lower_r and idx[1] < upper_r and idx[2] > lower_l and idx[2] < upper_l]
        self.one_idxs = [idx for idx in one_idxs if idx[1] > lower_r and idx[1] < upper_r and idx[2] > lower_l and idx[2] < upper_l]

    def generate_sample_locations(self, batch_size):
        # generate locations half from the positive set and half from the negative set
        p_locations = random.sample(self.zero_idxs, min(len(self.zero_idxs), batch_size // 2))
        n_locations = random.sample(self.one_idxs, min(len(self.one_idxs), batch_size - batch_size // 2))
        locations = np.vstack([p_locations, n_locations])
        return locations

class Uniform_Sampler:
    def __init__(self, lbls, msks, patch_size):
        self.lbls = np.squeeze(lbls, 3)
        self.msks = np.squeeze(msks, 3)
        sample_idxs = np.asarray(np.where(self.msks !=0)).T.tolist()
        self.patch_size = patch_size

        lower_l = patch_size[0]//2 
        upper_l = self.lbls.shape[1]-patch_size[0]//2
        lower_r = patch_size[1]//2 
        upper_r = self.lbls.shape[2]-patch_size[1]//2
        self.sample_idxs = [idx for idx in sample_idxs if idx[1] > lower_r and idx[1] < upper_r and idx[2] > lower_l and idx[2] < upper_l]

    def generate_sample_locations(self, batch_size):
        threshold = 0.45
        sample_list = np.zeros((batch_size, 3))

        counter = 0
        rejected = 0

        while counter < batch_size:
            # Randomly sample a centre point 
            [img, x, y] = random.sample(self.sample_idxs, 1)[0]

            # Count the class occurances within the sample
            sample = self.lbls[img, x-self.patch_size[0]//2:x+self.patch_size[0]-self.patch_size[0]//2, y-self.patch_size[1]//2:y+self.patch_size[1]-self.patch_size[1]//2]
            _, counts = np.unique(sample, return_counts=True) 
            relative_counts = counts/(self.patch_size[0]*self.patch_size[1])

            # Accept the sample if the class occurances are lower than the threshold  
            if np.max(relative_counts) < threshold : 
                sample_list[counter, :] = [img, x, y]
                counter += 1
            
            else:
                rejected += 1
            
            # Adjust the threshold if not enough samples can be found
            if rejected == (counter+1)*4: 
                threshold += 0.1
                rejected = 0

        return sample_list