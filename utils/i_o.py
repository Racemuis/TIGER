# Imports
import os
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(r'C:\Program Files\ASAP 2.0\bin')
import multiresolutionimageinterface as mir

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


def png_to_numpy(path: Path) -> np.array:
    """Opens a multiresolution png image with pillow

    Args:
        path (Path): path to image

    Raises:
        FileNotFoundError: raises when opened image is not found

    Returns:
        np.array: opened multiresolution image
    """
    img_frame = Image.open(path).convert('RGB')
    return np.array(img_frame, dtype=int)
"""
def process_folder(path: Path, targets: np.array = None, level: int = 5) -> np.array:
    Opens all images in a folder and stores them in a np.array

    Args:
        path (Path): path to image folder
        targets (np.array): An array of the files that should be read from folder; if None, all files are read
            Default: None
        level (int): the zoom level of the tif image (lower = heavier)
    
    Raises:
         FileNotFoundError: raises when opened folder is not found
    
    Returns:
        np.array: all the images in the directory

    TODO:   Add possibility to read XML files for the labels
    
    # The valid image types
    valid_images = [".png",".tif"]

    img_dir = []

    # Selecting which files to read
    files = targets if targets is not None else os.listdir(path)
    x_dir = os.listdir(path)
    x = len(x_dir)
    kind = os.path.splitext(files[0])[1]
    
    if kind == ".png": 
        images_dir = np.array([])

    elif kind == ".tif":
        # Reading of tif files     
        reader = mir.MultiResolutionImageReader()
        image = reader.open(os.path.join(path, x_dir[0]))
        width, height = image.getLevelDimensions(level)
        image_array = image.getUCharPatch(startX=0, startY=0, width=width, height=height, level=level)
        y, z, c = image_array.shape
        images_dir = np.zeros((x, y, z, c))

    for i, img in enumerate(files):
        ext = os.path.splitext(img)[1]
        if ext.lower() not in valid_images:
            print(f"The file {img} has not got a valid extension.\nValid extensions are: .png, .tif", file=sys.stderr)
            continue

        elif ext == ".png":
            np.append(images_dir, png_to_numpy(os.path.join(path, img)))

        elif ext == ".tif":
            images_dir[i, :, :, :] = tif_to_numpy(os.path.join(path, img), level = level, width = width, height = height)

    return images_dir.astype(int)"""

def process_folder(path: Path, targets: np.array = None, level: int = 5) -> np.array:
    """Opens all images in a folder and stores them in a np.array

    Args:
        path (Path): path to image folder
        targets (np.array): An array of the files that should be read from folder; if None, all files are read
            Default: None
        level (int): the zoom level of the tif image (lower = heavier)
    
    Raises:
         FileNotFoundError: raises when opened folder is not found
    
    Returns:
        np.array: all the images in the directory

    TODO:   Add possibility to read XML files for the labels
    """
    # The valid image types
    valid_images = [".png",".tif"]

    img_dir = []

    # Selecting which files to read
    files = targets if targets is not None else os.listdir(path)
    
    for img in files:
        ext = os.path.splitext(img)[1]
        if ext.lower() not in valid_images:
            print(f"The file {img} has not got a valid extension.\nValid extensions are: .png, .tif", file=sys.stderr)
            continue

        elif ext == ".png":
            img_dir.append(png_to_numpy(os.path.join(path, img)))

        elif ext == ".tif":
            img_dir.append(tif_to_numpy(os.path.join(path, img), level = level))

    return img_dir