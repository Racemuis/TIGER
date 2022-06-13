import os 
import sys
import json
import numpy as np 
from pathlib import Path
from PIL import Image

from tqdm import tqdm

# Celena directory
#X_DIR = r'C:\Users\celen\Documents\Radboud year 1\Intelligent Systems in Medical Imaging\TIGER\data_sample\wsirois\roi-level-annotations\tissue-cells\images'
#y_DIR = r'C:\Users\celen\Documents\Radboud year 1\Intelligent Systems in Medical Imaging\TIGER\data_sample\wsirois\roi-level-annotations\tissue-cells\tiger-coco.json'

#Chiara directory
X_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data\wsirois\roi-level-annotations\tissue-cells'
Y_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsirois\roi-level-annotations\tissue-cells\tiger-coco.json'

def get_images(path):
    with open(path, 'r') as file:
        data = json.load(file)
        return data['images']

def load_img(path: Path, level: int = 5)-> np.array:
    valid_images = [".png",".tif"]
    ext = os.path.splitext(path)[1]

    if ext.lower() not in valid_images:
        print(f"The file {path} has not got a valid extension.\nValid extensions are: .png, .tif", file=sys.stderr)
        return None

    elif ext == ".png":
        return np.array(Image.open(path))

def reshape(img: np.array, dim = 1024) -> np.array:
    x, y, _ = img.shape
    img_cropped = img[:min(x,dim), :min(y,dim), :]
    padding = ((0,max(0,dim-x)),(0,max(0,dim-y)),(0,0))
    return np.pad(img_cropped, pad_width = padding, mode='constant', constant_values = 0)

images = get_images(Y_DIR)

for img in tqdm(images, desc="Cropping images"): 
    image  = load_img(os.path.join(X_DIR, img['file_name']))
    im = Image.fromarray(reshape(image))
    im.save(fr"C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data\wsirois\roi-level-annotations\tissue-cells\cropped_images\{img['id']}.png")