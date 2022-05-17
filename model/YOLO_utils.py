# tensorflow as base library for neural networks
import tensorflow as tf
import numpy as np
import matplotlib.patches as patches

# keras as a layer on top of tensorflow
from keras.models import Model
from keras.layers import Reshape, Conv2D, MaxPooling2D, BatchNormalization, Lambda, LeakyReLU, concatenate

from typing import Union
from pathlib import Path
import cv2  

import requests
import zipfile
from typing import Union
from tqdm.notebook import tqdm


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


def xml2bbox(xml_file):
    '''Convert xml file to bounding box
    
    Parameters
    ----------
    xml_file: str
        Path to xml file
        
    Returns
    -------
    bbox: BoundingBox object
        A BoundingBox object with parameters derived from the xml file
    '''
    # Parse structure of XML file
    tree = ET.parse(xml_file)

    for elem in tree.iter():
        if 'width' in elem.tag:
            width = int(elem.text) # image width
        if 'height' in elem.tag:
            height = int(elem.text) # image height

        for attr in list(elem):

            if 'bndbox' in attr.tag:
                for dim in list(attr):
                    if 'xmin' in dim.tag:
                        xmin = int(dim.text) # xmin bounding box
                    if 'ymin' in dim.tag:
                        ymin = int(dim.text) # ymin bounding box
                    if 'xmax' in dim.tag:
                        xmax = int(dim.text) # xmax bounding box
                    if 'ymax' in dim.tag:
                        ymax = int(dim.text) # ymax bounding box

    x = (xmin+(xmax - xmin)/2)/width
    y = (ymin+(ymax - ymin)/2)/height
    w = (xmax - xmin)/width
    h = (ymax - ymin)/height
        
    return BoundingBox(x,y,w,h)

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
    for box in boxes:
        xmin  = int((box.x - box.w/2) * img_shape[1])
        xmax  = int((box.x + box.w/2) * img_shape[1])
        ymin  = int((box.y - box.h/2) * img_shape[0])
        ymax  = int((box.y + box.h/2) * img_shape[0])        
        plt_boxes.append(patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, color='#00FF00', linewidth='2'))
    return plt_boxes

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
    img_shape = image.shape
    input_image = np.expand_dims(input_image, 0) 

    # define variable needed to process input image
    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))

    # get output from network
    netout = model.predict([input_image, dummy_array])

    return decode_netout(netout[0], obj_threshold, nms_threshold, anchors, nb_class)  
   