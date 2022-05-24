import numpy as np
import random

class BalancedSampler:

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
