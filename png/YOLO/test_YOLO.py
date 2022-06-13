import cv2
import os
import json
import numpy as np
from tensorflow import keras

import matplotlib.pyplot as plt
from matplotlib import patches

from dependencies.decoding import predict_bounding_box

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

LABELS = ["lymphocytes and plasma cells"]

IMAGE_H, IMAGE_W = 1024, 1024 # use nearest power of 2 size, orginal height 1253, width 1326
GRID_H,  GRID_W  = 32 , 32
BOX              = 5
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD    = 0.3
NMS_THRESHOLD    = 0.3
ANCHORS          = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

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
X_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsirois\roi-level-annotations\tissue-cells\images'
y_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsirois\roi-level-annotations\tissue-cells\tiger-coco.json'

all_imgs, seen_labels = parse_annotation(y_DIR, X_DIR, ["lymphocytes and plasma cells"])

input_image = keras.layers.Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes  = keras.layers.Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

wt_path = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\cluster_YOLO.h5'
model = keras.models.load_model(wt_path)
print("Weights loaded from disk")

def get_matplotlib_boxes(boxes, img_shape):
    plt_boxes = []
    for box in boxes:
        xmin  = int((box.x - box.w/2) * img_shape[1])
        xmax  = int((box.x + box.w/2) * img_shape[1])
        ymin  = int((box.y - box.h/2) * img_shape[0])
        ymax  = int((box.y + box.h/2) * img_shape[0])        
        plt_boxes.append(patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, color='#00FF00', linewidth='2'))
    return plt_boxes

# define a threshold to apply to predictions
obj_threshold=0.01

test_img = all_imgs[4]['filename']

boxes, img_shape = predict_bounding_box(test_img, model, obj_threshold, NMS_THRESHOLD, ANCHORS, CLASS, TRUE_BOX_BUFFER, IMAGE_H, IMAGE_W)

print(boxes)

# get matplotlib bbox objects
plt_boxes = get_matplotlib_boxes(boxes, img_shape)

# visualize result
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, aspect='equal')
plt.imshow(cv2.imread(str(test_img)), cmap='gray')
for plt_box in plt_boxes:
    ax.add_patch(plt_box)
plt.show()