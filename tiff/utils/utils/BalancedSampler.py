import numpy as np
import random

class BalancedSampler:

    def __init__(self, lbls, msks):
        # pre calculate the positive and negative indices
        #lbls = np.squeeze(lbls, 3)
        #msks = np.squeeze(msks, 3)
        p_idxs = []
        n_idxs = []

        for lbl, msk in zip(lbls, msks):
            p_idxs.append(np.asarray(np.where((lbl.squeeze()) > 0)).T.tolist())
            n_idxs.append(np.asarray(np.where((msk.squeeze() > 0) & ~(lbl.squeeze() > 0))).T.tolist())

        self.p_idxs = p_idxs
        self.n_idxs = n_idxs
        # self.p_idxs = np.asarray(np.where(lbls > 0)).T.tolist()
        # self.n_idxs = np.asarray(np.where((msks > 0) & ~(lbls > 0))).T.tolist()

    def generate_sample_locations(self, batch_size):
        # generate locations half from the positive set and half from the negative set
        p_locations = np.zeros((len(self.p_idxs) * (batch_size//2), 3))
        n_locations = np.zeros((len(self.n_idxs) * (batch_size - batch_size//2), 3))

        for i, img_p in enumerate(self.p_idxs):
            for j in range(batch_size//2):
                p_locations[i*(batch_size//2)+j, :] = np.array([i, *random.sample(img_p, 1)[0]])
        for i, img_n in enumerate(self.n_idxs):
            for j in range(batch_size - batch_size//2):
                n_locations[i*(batch_size - batch_size//2)+j, :] = np.array([i, *random.sample(img_n, 1)[0]])
        
        locations = np.vstack((p_locations, n_locations)).astype(int)
        return locations

