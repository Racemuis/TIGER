
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import multiresolutionimageinterface as mir


from model.YOLO_utils import *
from utils.train_test_split import clean_train_test_split
from utils.i_o import process_folder
from utils.DataSet import DataSet
from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.annotation.structures import Point

class ROI:
    def __init__(self, shape, img_patch) -> None:
        self.img_patch = img_patch
        self.shape = shape
        self.annotations = []

    def add_annotation(self, annotation):
        self.annotations.append(annotation)

    def get_matplotlib_boxes(self, annotations):
        plt_boxes = []
        [min_x, min_y, _, _] = self.shape.bounds
        for a in annotations: 
            box = np.array(a.coordinates[:-1])
            x_min = np.amin(box, axis = 0)[0]-min_x
            x_max = np.amax(box, axis = 0)[0]-min_x
            y_min = np.amin(box, axis = 0)[1]-min_y
            y_max = np.amax(box, axis = 0)[1]-min_y
            plt_boxes.append(patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, fill=False, color='#00FF00', linewidth='1'))
        return plt_boxes

    def plot(self, title):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, aspect='equal')
        plt.imshow(self.img_patch, cmap='gray')
        plt.title(title)
        plt_boxes = self.get_matplotlib_boxes(self.annotations)
        for plt_box in plt_boxes:
            ax.add_patch(plt_box)
        plt.show()

# Set paths to the data
path = os.getcwd()
parent = os.path.dirname(path)

X_DIR = os.path.join(parent,r'data_sample\wsirois\wsi-level-annotations\images')
y_DIR = os.path.join(parent,r'data_sample\wsirois\wsi-level-annotations\annotations-tissue-cells-xmls')
MSKS_DIR =os.path.join(parent, r'data_sample\wsirois\wsi-level-annotations\annotations-tissue-cells-masks')

# Create a train-test split
X_train_files, X_test_files, y_train_files, y_test_files, _ , _  = clean_train_test_split(X_directory=X_DIR, y_directory= y_DIR, test_size=0.4, shuffle=False)

# Read the data
X_train = process_folder(path = X_DIR, targets=X_train_files, level = 5)
y_train = process_folder(path=y_DIR, targets=y_train_files)
msks_train = process_folder(path = MSKS_DIR, targets = X_train_files)

train_set = DataSet(X_train, msks_train, y_train)

X_test = process_folder(path= X_DIR, targets = X_test_files)
y_test = process_folder(path=y_DIR, targets=y_test_files)
msks_test = process_folder(path = MSKS_DIR, targets = X_test_files)

test_set = DataSet(X_test, msks_test, y_test)

# Define bounding boxes for all xml files
reader = mir.MultiResolutionImageReader()
rois = []
for data_file, XML_file in zip(os.listdir(X_DIR), os.listdir(y_DIR)): 
    XML_file_name = os.path.join(y_DIR, XML_file)
    data_file_name = os.path.join(X_DIR, data_file)

    image = reader.open(str(data_file_name))
    if image is None:
        raise IOError(f"Error opening image: {data_file_name}, image is None")

    # Get the width and the height of the image
    width, height = image.getLevelDimensions(0)

    # Get the centre of the image
    x = width//2
    y = height//2

    # Read the XML flile
    wsa = WholeSlideAnnotation(XML_file_name)
    annotations = wsa.select_annotations(x, y, width, height)

    # Select the roi's
    for a in annotations:
        if a.label.name == "roi":
            [min_x, min_y, max_x, max_y] = a.bounds
            rois.append(ROI(a, image.getUCharPatch(startX=min_x, startY=min_y, width=max_x-min_x, height=max_y-min_y, level=0)))


    # Select the annotations that represent lymphocytes per roi
    for a in annotations:
        if a.label.name == "lymphocytes and plasma cells":
            for roi in rois:
                if roi.shape.contains(a):
                    roi.add_annotation(a)


# Visualize region of interests and bounding boxes
for i, roi in enumerate(rois):
    print(roi.img_patch.shape)
    #roi.plot(title = f"Image {int(np.floor(i/3))+1} - region of interest {(i+1)%4}")


"""
def pad(img, new_width, new_height):
        """
        image: list of images (numpy arrays)
        returns a padded version of the image, such that the shape is equal to new_width and new_height
        """
        pad_width = new_width - img['width']
        pad_height = new_height - img['height']

        # Don't update the image yet because then we get tissue that is not inside the ROI
        # img['min_x'] = img['min_x'] - pad_width//2
        # img['max_x'] = img['max_x'] + (pad_width - pad_width//2)
        # img['min_y'] = img['min_y'] - pad_height//2
        # img['max_y'] = img['max_y'] + (pad_height - pad_height//2) 

        # Update annotations - Check for correctness!
        for annotation in img['object']:
            annotation['xmin'] = annotation['xmin'] - pad_width//2
            annotation['xmax'] = annotation['xmax'] - pad_width//2
            annotation['ymin'] = annotation['ymin'] - pad_height//2
            annotation['ymax'] = annotation['ymax'] - pad_height//2


        return img #np.pad(np.array(images), pad_width=paddings, mode='constant') 
"""

