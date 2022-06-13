# Adapted from Intelligent systems in medical imaging - Assignment 6

import tensorflow as tf
import numpy as np
import cv2

from pathlib import Path
from typing import Union

from dependencies.BoundBox import BoundBox, bbox_iou, tf_iou

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1, t=-100.0):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x / np.min(x) * t
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)

def decode_netout(netout, obj_threshold, nms_threshold, anchors, nb_class):
    """
        Decode output tensor of YOLO network and return list of BoundingBox objects.
    """
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []
    
    # decode the output by the network
    netout[..., 4]  = sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]
                
                if classes.any():
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + sigmoid(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]
                    
                    box = BoundBox(x, y, w, h, confidence, classes)
                    
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
                    
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
                        
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    
    return boxes

def predict_bounding_box(img: Union[Path, str], model, obj_threshold, nms_threshold, anchors, nb_class, TRUE_BOX_BUFFER, IMAGE_H, IMAGE_W):
    """
        Predict bounding boxes for a given image.
    """    
    image = cv2.imread(str(img))
    input_image = image / 255. # rescale intensity to [0, 1]
    input_image = input_image[:IMAGE_H,:IMAGE_W,::-1]
    img_shape = image.shape
    input_image = np.expand_dims(input_image, 0) 

    # define variable needed to process input image
    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))

    # get output from network
    netout = model.predict([input_image, dummy_array])

    return decode_netout(netout[0], obj_threshold, nms_threshold, anchors, nb_class), img_shape   