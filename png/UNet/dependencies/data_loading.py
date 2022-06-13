import os
import sys
import numpy as np
import multiresolutionimageinterface as mir

from PIL import Image
from pathlib import Path


def get_file_list(path: Path, ext: str=''):
    """Get the absolute paths to all files in a directory 
        that have the extension [ext].

    Args:
        path (Path): path to the directory
        ext (str): The file extension

    Returns:
        List: The sorted absolute paths to the files from the directory
    """
    return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)])

def get_contents(path: Path, ext: str =''):
    """Get the filenames of all files in a directory 
        that have the extension [ext].

    Args:
        path (Path): path to the directory
        ext (str): The file extension

    Returns:
        List: The sorted filenames from the directory
    """
    return sorted(f for f in os.listdir(path) if f.endswith(ext))

def tif_to_numpy(path: Path, level: int, width: int = None, height: int = None) -> np.array:
    """Opens a multiresolution tif image with ASAP python bindings

    Args:
        path (Path): path to image
        level (int): the zoom level of the image (lower = heavier)
        width (int): the width of the image patch, if None, the entire image is selected
            Default = None
        height (int): the height of the image patch, if None, the entire image is selected
            Default = None


    Raises:
        IOError: raises when opened image is None

    Returns:
        np.array: opened multiresolution image as a numpy array
    """
    reader = mir.MultiResolutionImageReader()
    image = reader.open(str(path))
    if image is None:
        raise IOError(f"Error opening image: {path}, image is None")

    # Get the width and the height of the image
    x, y = image.getLevelDimensions(level)

    if width == None:
        width = x
    if height == None:
        height = y

    # Convert a patch of the image to an numpy array
    image_array = image.getUCharPatch(startX=0, startY=0, width=width, height=height, level=level)
    return image_array.astype(int)

def load_img(path: Path, level: int = 5)-> np.array:
    """Load an image into a numpy array.
        Supported images: [.tif , .png]

    Args:
        path (Path): path to image
        level (int): the zoom level of the image (lower = heavier)
    Returns:
        np.array: The image as numpy array.
    """
    valid_images = [".png",".tif"]
    ext = os.path.splitext(path)[1]

    if ext.lower() not in valid_images:
        print(f"The file {path} has not got a valid extension.\nValid extensions are: .png, .tif", file=sys.stderr)
        return None

    elif ext == ".png":
        return np.array(Image.open(path))

    elif ext == ".tif":
        tif_to_numpy(level = level)

def reshape(img: np.array, dim = 512) -> np.array:
    """Pad or crop the image to match the given dimension

    Args:
        img (np.array): the original image
        dim (int): the new image width and height

    Returns:
        np.array: The padded image to [dim, dim, 3].
    """
    x, y, _ = img.shape
    img_cropped = img[:min(x,dim), :min(y,dim), :]
    padding = ((0,max(0,dim-x)),(0,max(0,dim-y)),(0,0))
    return np.pad(img_cropped, pad_width = padding, mode='constant', constant_values = 0)