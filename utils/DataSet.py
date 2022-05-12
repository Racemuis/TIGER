import matplotlib.pyplot as plt

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
                ax.imshow(im, cmap='gray')
                ax.set_title(t)
        else:
            f, axes = plt.subplots(1, 2)
            for ax, im, t in zip(axes, 
                                 (self.imgs[i], self.msks[i]), 
                                 ('RGB image', 'Mask')):
                ax.imshow(im, cmap='gray')
                ax.set_title(t)
        plt.show()