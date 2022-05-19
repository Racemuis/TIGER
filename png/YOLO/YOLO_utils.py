# tensorflow as base library for neural networks
import tensorflow as tf
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from os import listdir
import json

# keras as a layer on top of tensorflow
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, LeakyReLU, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from pathlib import Path
import cv2  

import requests
import zipfile
from typing import Union
from tqdm.notebook import tqdm

from loss_utils import *
import utils as utils
from preprocessing import parse_annotation, BatchGenerator, normalize


def download_weights(url: str,
                           download_zipfile_name: Union[Path,str],
                           extract_dir_name: Union[Path,str],
                           workdir: Union[Path,str] = ".") -> None:
    
    workdir = Path(workdir)
    download_zipfile_name = workdir / download_zipfile_name
    extract_dir_name = workdir / extract_dir_name
    
    if not download_zipfile_name.is_file():
        with open(download_zipfile_name, "wb") as f:
                response = requests.get(url, stream=True)
                total_length = response.headers.get('content-length')
                if total_length is None: # no content length header
                    f.write(response.content)
                else:
                    dl = 0
                    chunk_size = 4096
                    total_length = int(total_length)
                    with tqdm(response.iter_content(chunk_size=chunk_size), total=total_length, desc='Downloading data') as pbar:
                        for data in pbar:
                            dlen = len(data) 
                            f.write(data)
                            dl += dlen
                            pbar.update(dlen)
    if not extract_dir_name.is_dir():   # caters for case when zip file downloaded but e.g. wasn't extracted
        with zipfile.ZipFile(download_zipfile_name,"r") as zip_ref:
            zip_ref.extractall(workdir)    

def json2bbox(annotation_file, image_id):
    '''Convert json file to bounding box
    
    Parameters
    ----------
    annotation_file: str
        Path to json file
    image_id: int
        Id of the image to analyze
        
    Returns
    -------
    bbox: BoundingBox object
        A BoundingBox object with parameters derived from the json file
    '''

    json_file = open(annotation_file)
    json_data = json.load(json_file)
    
    width = json_data['images'][image_id]['width']
    height = json_data['images'][image_id]['height']
    
            
    n_boxes = []
    
    for annotations in json_data['annotations']:
        if annotations['image_id'] == image_id:
            
            x = annotations['bbox'][0]
            y = annotations['bbox'][1]
            w = annotations['bbox'][2]
            h = annotations['bbox'][3]
            n_boxes.append(BoundingBox(x, y, w, h))
        
        else:
            continue

        
    return n_boxes

class BoundingBox:
    def __init__(self, x, y, w, h, c = None, classes = None):
        """A bounding box object.
        
        Parameters
        ----------
        x: float
            x coordinate of the center of the box
        y: float
            y coordinate of the center of the box
        w: float
            width of the box
        h: float
            height of the box
        """
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score
    
def get_matplotlib_boxes(boxes, img_shape):
    
    plt_boxes = []
    plt_coords = []
    for box in boxes:
        for b in box:
            xmin  = int(b.x)
            xmax  = int(b.x + b.w)
            ymin  = int(b.y)
            ymax  = int(b.y + b.h)        
            plt_boxes.append(patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, color='#00FF00', linewidth='2'))
            plt_coords.append([xmin, xmax, ymin, ymax])
    return plt_boxes, plt_coords


def space_to_depth_x2(x):
    return tf.nn.space_to_depth(x, block_size=2)

def YOLO_network(input_img, true_bxs, CLASS, BOX, GRID_H, GRID_W):

    # Layer 1
    x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_img)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
    x = BatchNormalization(name='norm_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
    x = BatchNormalization(name='norm_3')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 4
    x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
    x = BatchNormalization(name='norm_4')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
    x = BatchNormalization(name='norm_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 7
    x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
    x = BatchNormalization(name='norm_7')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 8
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False, input_shape=(512,512,3))(x)
    x = BatchNormalization(name='norm_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
    x = BatchNormalization(name='norm_9')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
    x = BatchNormalization(name='norm_10')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
    x = BatchNormalization(name='norm_11')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
    x = BatchNormalization(name='norm_12')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
    x = BatchNormalization(name='norm_13')(x)
    x = LeakyReLU(alpha=0.1)(x)

    skip_connection = x

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
    x = BatchNormalization(name='norm_14')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
    x = BatchNormalization(name='norm_15')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
    x = BatchNormalization(name='norm_16')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
    x = BatchNormalization(name='norm_17')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 18
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
    x = BatchNormalization(name='norm_18')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
    x = BatchNormalization(name='norm_19')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
    x = BatchNormalization(name='norm_20')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 21
    skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
    skip_connection = BatchNormalization(name='norm_21')(skip_connection)
    skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
    skip_connection = Lambda(space_to_depth_x2)(skip_connection)

    x = concatenate([skip_connection, x])

    # Layer 22
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
    x = BatchNormalization(name='norm_22')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 23
    x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
    output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

    # small hack to allow true_boxes to be registered when Keras build the model 
    # for more information: https://github.com/fchollet/keras/issues/2790
    output = Lambda(lambda args: args[0])([output, true_bxs])

    model = Model([input_img, true_bxs], output)
    
    model.summary()
    
    return model

def decode_netout(netout, obj_threshold, nms_threshold, anchors, nb_class):
    """
        Decode output tensor of YOLO network and return list of BoundingBox objects.
    """
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []
    
    # decode the output by the network
    netout[..., 4]  = utils.sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * utils.softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]
                
                if classes.any():
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + utils.sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + utils.sigmoid(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]
                    
                    box = BoundingBox(x, y, w, h, confidence, classes)
                    
                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if utils.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
                        
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    
    return boxes

def predict_bounding_box(img: Union[Path, str], model, obj_threshold, nms_threshold, anchors, nb_class, TRUE_BOX_BUFFER):
    """
        Predict bounding boxes for a given image.
    """    
    image = cv2.imread(str(img))
    image.resize(256,256,3)
    input_image = image / 255. # rescale intensity to [0, 1]
    input_image = input_image[:,:,::-1]
    input_image = np.expand_dims(input_image, 0) 

    # define variable needed to process input image
    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))

    # get output from network
    netout = model.predict([input_image, dummy_array])

    return decode_netout(netout[0], obj_threshold, nms_threshold, anchors, nb_class)  

class Loss(tf.keras.losses.Loss):

    EPSILON = 1e-6

    def __init__(
        self,
        batch_size,
        grid_width,
        grid_height,
        anchors,
        n_boxes=1000,
        scales=Scales()
    ):

        self._batch_size = batch_size
        self._grid_width = grid_width
        self._grid_height = grid_height

        self._n_boxes = n_boxes
        self._anchors = anchors
        self._scales = scales

        self._grid = get_cell_grid(
            self._batch_size, self._n_boxes, self._grid_width, self._grid_height
        )
        
        self.reduction = tf.keras.losses.Reduction.AUTO
        self.name = 'loss'

    def __call__(self, y_true, y_pred, sample_weight=None):

        # prediction
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + self._grid
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(
            self._anchors, [1, 1, 1, self._n_boxes, 2]
        )
        pred_box_conf = tf.sigmoid(y_pred[..., 4])
        pred_box_class = y_pred[..., 5:]

        # ground thuth
        true_box_xy = y_true[..., :2]
        true_box_wh = y_true[..., 2:4]
        true_box_conf = (
            tf_iou(true_box_xy, true_box_wh, pred_box_xy, pred_box_wh) * y_true[..., 4]
        )
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        # expand for  2 values (x,y) and (w,h) for coord scale
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self._scales.coord_scale
        conf_mask = (y_true[..., 4] + self._scales.no_object_scale) * self._scales.object_scale
        class_mask = y_true[..., 4] * self._scales.class_scale

        # normalization factor
        nb_coord_norm = tf.reduce_sum(tf.cast(coord_mask > 0.0, dtype=tf.float32)) + Loss.EPSILON
        nb_conf_norm = tf.reduce_sum(tf.cast(conf_mask > 0.0, dtype=tf.float32)) + Loss.EPSILON
        nb_class_norm = tf.reduce_sum(tf.cast(class_mask > 0.0, dtype=tf.float32)) + Loss.EPSILON

        # losses
        loss_xy = (
            tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask)
            / nb_coord_norm
            / 2.0
        )
        loss_wh = (
            tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask)
            / nb_coord_norm
            / 2.0
        )
        loss_conf = (
            tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask)
            / nb_conf_norm
            / 2.0
        )
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=true_box_class, logits=pred_box_class
        )
        loss_class = tf.reduce_sum(loss_class * class_mask) / nb_class_norm

        # combine all terms
        loss = loss_xy + loss_wh + loss_conf + loss_class
        return loss


####
####
####
####
####  Calling functions
####
####
####
####

def reshape(img: np.array, dim = 256) -> np.array:
    x, y, _ = img.shape
    img_cropped = img[:min(x,dim), :min(y,dim), :]
    padding = ((0,max(0,dim-x)),(0,max(0,dim-y)),(0,0))
    return np.pad(img_cropped, pad_width = padding, mode='constant')

workdir = Path(".")

# Define directory of training images
train_dir_images = workdir / 'roi-level-annotations' / 'tissue-cells' /'images'
annotation_file = workdir / 'roi-level-annotations' / 'tissue-cells' /'tiger-coco.json'

imgs = []

print('Reading images and labels...')
for image_id in range(len(listdir(train_dir_images))):
    
    case_dict = {}
    
    case = train_dir_images / (listdir(train_dir_images)[image_id])
    
    # Open image with opencv and visualize it
    image = cv2.imread(str(case))
    image = reshape(image)
    bbox = json2bbox(annotation_file, image_id)
    
    case_dict['id'] = image_id
    case_dict['filename'] = str(case)
    case_dict['image'] = image
    case_dict['bbox'] = bbox
    
    imgs.append(case_dict)
    
    if image_id == 200:
        break
    
print('Images and labels loaded')
'''
id = 3

# Get bounding boxes in matplotlib format
plt_boxes = get_matplotlib_boxes([imgs[id]['bbox']], imgs[id]['image'].shape)

# Visualize image and bounding box
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, aspect='equal')
plt.imshow(imgs[id]['image'].squeeze(), cmap='gray')

for plt_box in plt_boxes:
    ax.add_patch(plt_box)
plt.show()
'''

# define configuration parameters for batch generator

LABELS = ['lymphocytes and plasma cells'] # Lym = lymphocites, Plm = plasma cells

CLASS           = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')

### Define YOLO params

IMAGE_H, IMAGE_W = 256, 256  #Shape of images
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

BATCH_SIZE       = 16
WARM_UP_BATCHES  = 100
TRUE_BOX_BUFFER  = 50

#Load config generator

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


input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

model = YOLO_network(input_image, true_boxes, CLASS, BOX, GRID_H, GRID_W)

# extract list of images and corresponding annotations

#all_imgs, _ = parse_annotation(annotation_file, train_dir_images)

# IMPORTANT: Please notice that we decided to use half of the dataset available in the 'train' folder
# to speed up the training process, you should do the same when training your own network in STEP 3.
#half_size =  len(all_imgs) // 2 # DO NOT MODIFY
#all_imgs = all_imgs[:half_size] # DO NOT MODIFY

# define percentage of data to use for training
training_data_percentage = 0.8 # set a number between 0 and 1

# define number of training images
train_valid_split = int(training_data_percentage * len(imgs))

# initialize training and validation batch generators
train_batch = BatchGenerator(imgs[:train_valid_split], generator_config, norm=utils.normalize)
valid_batch = BatchGenerator(imgs[train_valid_split:], generator_config, norm=utils.normalize)

   
early_stop = EarlyStopping(monitor='val_loss', 
                       min_delta=0.001, 
                       patience=10, 
                       mode='min', 
                       verbose=1)

checkpoint = ModelCheckpoint('pretrained_yolo_weights.h5',
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min')

# define number of epochs
n_epoch = 2
learning_rate = 1e-5

# define a folder where to store log files during training
logwrite = workdir / 'logs'
logwrite.mkdir(exist_ok=True)

# Define Adam optimizer
optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.75, epsilon=1e-08, decay=0.0)

dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))

print('Training model...')

img_train = []
bbox_train = []

for image in range(len(imgs[:train_valid_split])):
    img_train.append(imgs[:train_valid_split][image]['image'])
    bbox_train.append(get_matplotlib_boxes([imgs[:train_valid_split][image]['bbox']], (256,256,3))[1])
    
img_valid = []
bbox_valid = []

for image in range(len(imgs[train_valid_split:])):
    img_valid.append(imgs[train_valid_split:][image]['image'])
    bbox_valid.append(get_matplotlib_boxes([imgs[train_valid_split:][image]['bbox']], (256,256,3))[1])
    
    
# compile YOLO model
loss = Loss(16, 16, 16, ANCHORS, n_boxes = 5)
model.compile(loss=loss, optimizer=optimizer)
# do training
model.fit(x = img_train,
          y = bbox_train,
          epochs=n_epoch,
            callbacks=[early_stop, checkpoint])

print('Training finished')



# read from test set
#test_dir = workdir / 'test_images'
#test_file_list = listdir(test_dir)
#img_index = np.random.randint(0, len(test_file_list))
#test_img = cv2.imread(str(test_dir.joinpath(test_file_list[img_index])))



# define a threshold to apply to predictions
obj_threshold=0.2

# predict bounding boxes using YOLO model2
boxes = predict_bounding_box(val / test_file_list[img_index], model, obj_threshold, NMS_THRESHOLD, ANCHORS, CLASS)
#print(test_file_list[img_index])
# get matplotlib bbox objects
plt_boxes = get_matplotlib_boxes(boxes, test_img.shape)

# visualize result
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, aspect='equal')
plt.imshow(test_img, cmap='gray')
for plt_box in plt_boxes:
    ax.add_patch(plt_box)
plt.show()