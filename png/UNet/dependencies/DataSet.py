import matplotlib.pyplot as plt
import numpy as np

import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
class DataSet:
    
    def __init__(self, imgs, msks, lbls=None):
        self.imgs = imgs
        self.msks = msks
        self.lbls = lbls
    
    def show_image(self, i):
        if self.lbls != None:
            f, axes = plt.subplots(1, 3)
            for ax, im, t in zip(axes, 
                                 (self.imgs[i], self.msks[i], self.lbls[i]), 
                                 ('RGB image', 'Mask','Manual annotation')):
                ax.imshow(im)

                ax.set_title(t)
        else:
            f, axes = plt.subplots(1, 2)
            for ax, im, t in zip(axes, 
                                 (self.imgs[i], self.msks[i]), 
                                 ('RGB image', 'Mask')):
                ax.imshow(im)
                ax.set_title(t)
        values = [0,1,2]
        labels = ['Stroma', 'invasive tumor', 'rest']
        cmap = get_cmap('viridis')

        colours = [cmap((i)*0.5) for i in values]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=colours[i], label=labels[i] ) for i in range(len(values)) ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(-0.7, -0.25), borderaxespad=0., ncol = 3)
        plt.show()