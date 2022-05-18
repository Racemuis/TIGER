import os
import numpy as np
from PIL import Image


def get_file_list(path, ext=''):
    return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)])

def load_img(path):
    return np.array(Image.open(path))

def reshape(img: np.array, dim = 3500) -> np.array:
    x, y, _ = img.shape
    img_cropped = img[:min(x,dim), :min(y,dim), :]
    padding = ((0,max(0,dim-x)),(0,max(0,dim-y)),(0,0))
    return np.pad(img_cropped, pad_width = padding, mode='constant')