import numpy as np
import random

class BalancedSampler:

    def __init__(self, lbls, msks):
        # pre calculate the positive and negative indices
        lbls = np.squeeze(lbls, 3)
        msks = np.squeeze(msks, 3)
        self.p_idxs = np.asarray(np.where(lbls > 3)).T.tolist()
        self.n_idxs = np.asarray(np.where((msks > 0) & ~(lbls > 3))).T.tolist()

    def generate_sample_locations(self, batch_size):
        # generate locations half from the positive set and half from the negative set
        p_locations = random.sample(self.p_idxs, min(len(self.p_idxs), batch_size // 2))
        n_locations = random.sample(self.n_idxs, min(len(self.n_idxs), batch_size - batch_size // 2))
        locations = np.vstack([p_locations, n_locations])
        return locations
