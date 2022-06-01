import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

sys.path.append(r'C:\Program Files\ASAP 2.0\bin') # Fill in your own path here

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import multiresolutionimageinterface as mir

from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as pltpatches

# keras as a layer on top of tensorflow
from keras.layers import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# matplotlib is needed to plot bounding boxes
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

# import dependencies
from dependencies.BatchGenerator import BatchGenerator
from dependencies.YOLO_network import YOLO_network
from dependencies.Loss import Loss
from dependencies.decoding import predict_bounding_box

def normalize(image):
    image = image / 255.
    return image

def plot_image_with_boxes(image, boxes):
    f, ax = plt.subplots()
    ax.imshow(image)
    # draw boundingboxes
    for box in boxes:
        # Create a Rectangle patch
        s = 256 / 32
        xc, yc, wc, hc = box[:4] * s
        x = xc - wc / 2
        y = yc - hc / 2
        rect = pltpatches.Rectangle((x, y), wc, hc, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()

def convert_ground_truth_yollo_output_to_boxes(y_patch):
    return y_patch[np.where(y_patch[..., 5])]

def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}

    # Read the JSON flile
    with open(ann_dir, 'r') as f:
        data = json.load(f)

    for image in data['images']:

        img = {"object": []}
        filename = os.path.split(image['file_name'])[-1]

        # Only process the images that are in the image folder
        if filename in os.listdir(img_dir):
            img["filename"] = os.path.join(img_dir,filename)
        
            img["width"] = image['width']
            img["height"] = image['height']
            img["id"] = image['id']

            for annotation in data["annotations"]:
                obj = {}
                if annotation['image_id'] == img['id']:
                    obj["name"] = annotation["category_id"]

                    if obj["name"] in seen_labels:
                        seen_labels[obj["name"]] += 1
                    else:
                        seen_labels[obj["name"]] = 1

                    if obj["name"] == 1:
                        img["object"] += [obj]

                    box = annotation['bbox']
                    obj["xmin"] = box[0]
                    obj["xmax"] = box[0]+ box[2]
                    obj["ymin"] = box[1]
                    obj["ymax"] = box[1] + box[3]

            #if len(img["object"]) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels

CLUSTER_MODE = False

# Set paths to the data
os.chdir(os.path.dirname(os.getcwd()))
os.chdir(os.path.dirname(os.getcwd()))
os.chdir(os.path.dirname(os.getcwd()))
path = os.path.dirname(os.getcwd())
#X_DIR =  os.path.join(path,r'project/data_sample\wsirois\roi-level-annotations\tissue-cells\images')
#y_DIR =  os.path.join(path,r'project/data_sample\wsirois\roi-level-annotations\tissue-cells\tiger-coco.json')
X_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsirois\roi-level-annotations\tissue-cells\images'
y_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsirois\roi-level-annotations\tissue-cells\tiger-coco.json'
all_imgs, seen_labels = parse_annotation(y_DIR, X_DIR, ["lymphocytes and plasma cells"])


max_height = max([img['height'] for img in all_imgs])+1
max_width =  max([img['width'] for img in all_imgs])+1

# These parameters are mostly for the detection of the right lung
# TODO: change parameters
LABELS = ["lymphocytes and plasma cells"]

IMAGE_H, IMAGE_W = 256, 256
if CLUSTER_MODE:
    # This should be made the nearest power of 2 and the grid below should be adapted
    IMAGE_H, IMAGE_W = max_height, max_width 
GRID_H,  GRID_W  = 8 , 8
BOX              = 5
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD    = 0.3
NMS_THRESHOLD    = 0.3
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

NO_OBJECT_SCALE  = 1.0 # lambda noobj
OBJECT_SCALE     = 5.0 # lambda obj
COORD_SCALE      = 1.0
CLASS_SCALE      = 1.0

BATCH_SIZE       = 2
WARM_UP_BATCHES  = 100
TRUE_BOX_BUFFER  = 50

generator_config = {
    'IMAGE_H'         : IMAGE_H, 
    'IMAGE_W'         : IMAGE_W,
    'GRID_H'          : GRID_H,  
    'GRID_W'          : GRID_W,
    'BOX'             : BOX,
    'LABELS'          : LABELS,
    'CLASS'           : len(LABELS),
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : 50,
}


early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=10, 
                           mode='min', 
                           verbose=1)

checkpoint = ModelCheckpoint('model_YOLO.h5',
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min')


input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

# define percentage of data to use for training
training_data_percentage = 0.8 # set a number between 0 and 1

# define number of training images
train_valid_split = int(training_data_percentage * len(all_imgs))

# initialize training and validation batch generators
train_batch = BatchGenerator(all_imgs[:train_valid_split], max_width, max_height, generator_config, norm=normalize)
valid_batch = BatchGenerator(all_imgs[train_valid_split:], max_width, max_height, generator_config, norm=normalize)

# define number of epochs
n_epoch = 2
learning_rate = 1e-5

# Define Adam optimizer
optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.75, epsilon=1e-08, decay=0.0)

dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))

model = YOLO_network(input_image, true_boxes, CLASS, BOX, GRID_H, GRID_W)

# compile YOLO model
loss = Loss(BATCH_SIZE, GRID_H, GRID_W, ANCHORS, n_boxes = 5)
model.compile(loss=loss, optimizer=optimizer)

# do training
model.fit(train_batch,
            validation_data = valid_batch,
            epochs=n_epoch,
            callbacks=[early_stop, checkpoint])

model.save('./YOLO')

def get_matplotlib_boxes(boxes, img_shape):
    plt_boxes = []
    for box in boxes:
        xmin  = int((box.x - box.w/2) * img_shape[1])
        xmax  = int((box.x + box.w/2) * img_shape[1])
        ymin  = int((box.y - box.h/2) * img_shape[0])
        ymax  = int((box.y + box.h/2) * img_shape[0])        
        plt_boxes.append(patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, color='#00FF00', linewidth='2'))
    return plt_boxes

if not CLUSTER_MODE:
    # define a threshold to apply to predictions
    obj_threshold=0.5

    test_img = all_imgs[train_valid_split]['filename']

    boxes, img_shape = predict_bounding_box(test_img, model, obj_threshold, NMS_THRESHOLD, ANCHORS, CLASS, TRUE_BOX_BUFFER)
    
    # get matplotlib bbox objects
    plt_boxes = get_matplotlib_boxes(boxes, img_shape)

    # visualize result
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal')
    plt.imshow(cv2.imread(str(test_img)), cmap='gray')
    for plt_box in plt_boxes:
        ax.add_patch(plt_box)
    plt.show()