
from turtle import width
import matplotlib.pyplot as plt
import numpy as np
import copy
import tensorflow as tf

import os
from pathlib import Path
from tensorflow.keras.utils import Sequence
import multiresolutionimageinterface as mir


from model.YOLO_utils import *
from utils.train_test_split import clean_train_test_split
from utils.i_o import process_folder
from utils.DataSet import DataSet
from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.annotation.structures import Point
from imgaug import augmenters as iaa

import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as pltpatches


# tensorflow as base library for neural networks
import tensorflow as tf

# keras as a layer on top of tensorflow
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.layers import concatenate
import keras.backend as K

# h5py is needed to store and load Keras models
import h5py
 
# matplotlib is needed to plot bounding boxes
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm.notebook import tqdm
import cv2
from pathlib import Path


@dataclass
class Scales:
    coord_scale = 1.0
    no_object_scale = 0.5
    object_scale = 5.0  
    class_scale = 1.0
    

def get_cell_grid(batch_size, n_boxes, grid_width, grid_height):
    cell_x =  tf.cast(tf.reshape(
            tf.tile(tf.range(grid_width), [grid_height]),
            (1, grid_height, grid_width, 1, 1),
        ), dtype=tf.float32) 
 

    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
    return tf.tile(tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, n_boxes, 1])


def tf_iou(true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
    true_wh_half = true_box_wh / 2.0
    true_mins = true_box_xy - true_wh_half
    true_maxes = true_box_xy + true_wh_half

    pred_wh_half = pred_box_wh / 2.0
    pred_mins = pred_box_xy - pred_wh_half
    pred_maxes = pred_box_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.0)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)
    return iou_scores


def tf_decode(netout, obj_threshold, nms_threshold):

    netout[..., 4] = tf.sigmoid(netout[..., 4])
    netout[..., 5] = netout[..., 4] * softmax(netout[..., 5])
    netout[..., 5] *= netout[..., 5] > obj_threshold

    indices = tf.where(netout[..., 5] > 0)
    netout = netout[indices]

    netout[..., :2] = tf.sigmoid(netout[..., :2])
    netout[..., 0] = netout[..., 0] + indices[1]
    netout[..., 1] = netout[..., 1] + indices[0]
    netout[..., 2] = 1.5 * tf.exp(netout[..., 2])
    netout[..., 3] = 1.5 * tf.exp(netout[..., 3])

    # Non maximum surpression
    netout = netout[
        tf.nn.top_k(netout[:, 4], k=tf.size(netout[:, 4]), sorted=True).indices
    ][::-1]
    for i in range(netout.shape[0]):
        for j in range(i + 1, netout.shape[0]):
            if (
                tf_iou(netout[i, :2], netout[i, 2:4], netout[j, :2], netout[j, 2:4])
                >= nms_threshold
            ):
                netout[j][5] = 0.0
    return netout[tf.where(netout[..., 5] > 0)]


def np_iou(true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
    true_wh_half = true_box_wh / 2.0
    true_mins = true_box_xy - true_wh_half
    true_maxes = true_box_xy + true_wh_half

    pred_wh_half = pred_box_wh / 2.0
    pred_mins = pred_box_xy - pred_wh_half
    pred_maxes = pred_box_xy + pred_wh_half

    intersect_mins = np.maximum(pred_mins, true_mins)
    intersect_maxes = np.minimum(pred_maxes, true_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.0)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = np.true_divide(intersect_areas, union_areas)
    return iou_scores


def np_decode(netout, obj_threshold, nms_threshold):
    netout[..., 4] = sigmoid(netout[..., 4])
    netout[..., 5] = netout[..., 4] * softmax(netout[..., 5])
    netout[..., 5] *= netout[..., 5] > obj_threshold

    indices = np.where(netout[..., 5] > 0)
    netout = netout[indices]

    netout[..., :2] = sigmoid(netout[..., :2])
    netout[..., 0] = netout[..., 0] + indices[1]
    netout[..., 1] = netout[..., 1] + indices[0]
    netout[..., 2] = 1.5 * np.exp(netout[..., 2])  # !!!!hard code
    netout[..., 3] = 1.5 * np.exp(netout[..., 3])  # !!!!!hard code

    # Non maximum surpression
    netout = netout[netout[:, 4].argsort()][::-1]
    for i in range(netout.shape[0]):
        for j in range(i + 1, netout.shape[0]):
            if (
                np_iou(netout[i, :2], netout[i, 2:4], netout[j, :2], netout[j, 2:4])
                >= nms_threshold
            ):
                netout[j][5] = 0.0
    return netout[np.where(netout[..., 5] > 0)]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1, t=-100.0):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x / np.min(x) * t
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)


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


def convert_mask_to_yollo_output(mask, grid_shape, n_boxes, number_of_classes, output_shape, bounding_box_size):
    yollo_label = np.zeros((*grid_shape, n_boxes, 4 + 1 + number_of_classes))

    classes = list(np.unique(mask))
    classes.remove(0)

    for _class in classes:
        transformed_points = np.where(mask == _class)
        new_points = list(zip(transformed_points[0], transformed_points[1]))

        for point in new_points:
            center_x = point[1] / (output_shape[0] / grid_shape[0])
            center_y = point[0] / (output_shape[1] / grid_shape[1])

            # skip points that are on the border and will be assigned to grid_cell out of range
            if center_x >= grid_shape[0] or center_y >= grid_shape[1]:
                continue

            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))

            # relative to grid cell
            center_w = bounding_box_size / (output_shape[0] / grid_shape[0])
            center_h = bounding_box_size / (output_shape[1] / grid_shape[1])
            box = [center_x, center_y, center_w, center_h]

            # TODO find best anchor
            anchor = 0

            # ground truth
            yollo_label[grid_y, grid_x, anchor, 0:4] = box
            yollo_label[grid_y, grid_x, anchor, 4] = 1.  # confidence
            yollo_label[grid_y, grid_x, anchor, 4 + int(_class)] = 1  # class

    return yollo_label


def convert_ground_truth_yollo_output_to_boxes(y_patch):
    return y_patch[np.where(y_patch[..., 5])]

class FitYolo():

    def __init__(self,
                 label_map,
                 output_shape,
                 grid_shape,
                 n_boxes,
                 bounding_box_size):

        self._output_shape = output_shape
        self._grid_shape = grid_shape
        self._n_boxes = n_boxes
        self._bounding_box_size = bounding_box_size
        self._number_of_classes = len(label_map)

    def __call__(self, x, y):
        yollo_label = convert_mask_to_yollo_output(y,
                                                   self._grid_shape,
                                                   self._n_boxes,
                                                   self._n_boxes,
                                                   self._output_shape,
                                                   self._bounding_box_size)
        return x, yollo_label
    
    def reset(self):
        pass

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


class BoundBox:
    def __init__(self, x, y, w, h, c = None, classes = None):
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


def bbox_iou(box1, box2):
    x1_min  = box1.x - box1.w/2
    x1_max  = box1.x + box1.w/2
    y1_min  = box1.y - box1.h/2
    y1_max  = box1.y + box1.h/2

    x2_min  = box2.x - box2.w/2
    x2_max  = box2.x + box2.w/2
    y2_min  = box2.y - box2.h/2
    y2_max  = box2.y + box2.h/2

    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])

    intersect = intersect_w * intersect_h

    union = box1.w * box1.h + box2.w * box2.h - intersect

    return float(intersect) / union

class BatchGenerator(Sequence):
    def __init__(self, images, width, height, config, shuffle=True, jitter=True, norm=None):
        self.generator = None
        self.reader = mir.MultiResolutionImageReader()
        self.images = images
        self.width = width
        self.height = height
        self.config = config

        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm

        self.counter = 0
        self.anchors = [
            BoundBox(0, 0, config["ANCHORS"][2 * i], config["ANCHORS"][2 * i + 1])
            for i in range(int(len(config["ANCHORS"]) // 2))
        ]

        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                # iaa.Fliplr(0.5), # horizontally flip 50% of all images
                # iaa.Flipud(0.2), # vertically flip 20% of all images
                # sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(
                    iaa.Affine(
                        # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                        # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                        # rotate=(-5, 5), # rotate by -45 to +45 degrees
                        # shear=(-5, 5), # shear by -16 to +16 degrees
                        # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                        # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                        # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )
                ),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf(
                    (0, 5),
                    [
                        # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf(
                            [
                                iaa.GaussianBlur(
                                    (0, 3.0)
                                ),  # blur images with a sigma between 0 and 3.0
                                iaa.AverageBlur(
                                    k=(2, 7)
                                ),  # blur image using local means with kernel sizes between 2 and 7
                                iaa.MedianBlur(
                                    k=(3, 11)
                                ),  # blur image using local medians with kernel sizes between 2 and 7
                            ]
                        ),
                        iaa.Sharpen(
                            alpha=(0, 1.0), lightness=(0.75, 1.5)
                        ),  # sharpen images
                        # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges
                        # sometimes(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0, 0.7)),
                        #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                        # ])),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),  # add gaussian noise to images
                        iaa.OneOf(
                            [
                                iaa.Dropout(
                                    (0.01, 0.1), per_channel=0.5
                                ),  # randomly remove up to 10% of the pixels
                                # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                            ]
                        ),
                        # iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add(
                            (-10, 10), per_channel=0.5
                        ),  # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply(
                            (0.5, 1.5), per_channel=0.5
                        ),  # change brightness of images (50-150% of original value)
                        iaa.LinearContrast(
                            (0.5, 2.0), per_channel=0.5
                        ),  # improve or worsen the contrast
                        # iaa.Grayscale(alpha=(0.0, 1.0)),
                        # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                    ],
                    random_order=True,
                ),
            ],
            random_order=True,
        )

        if shuffle:
            np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images)) / self.config["BATCH_SIZE"]))

    def __getitem__(self, idx):
        l_bound = idx * self.config["BATCH_SIZE"]
        r_bound = (idx + 1) * self.config["BATCH_SIZE"]

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config["BATCH_SIZE"]

        instance_count = 0

        x_batch = np.zeros(
            (r_bound - l_bound, self.config["IMAGE_H"], self.config["IMAGE_W"], 3)
        )  # input images
        b_batch = np.zeros(
            (r_bound - l_bound, 1, 1, 1, self.config["TRUE_BOX_BUFFER"], 4)
        )  # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros(
            (
                r_bound - l_bound,
                self.config["GRID_H"],
                self.config["GRID_W"],
                self.config["BOX"],
                4 + 1 + self.config["CLASS"],
            )
        )  # desired network output

        for train_instance in self.images[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self.aug_image(train_instance, jitter=self.jitter)

            # construct output from object's x, y, w, h
            true_box_index = 0

            for obj in all_objs:
                if (
                    obj["xmax"] > obj["xmin"]
                    and obj["ymax"] > obj["ymin"]
                    and obj["name"] in self.config["LABELS"]
                ):
                    center_x = 0.5 * (obj["xmin"] + obj["xmax"])
                    center_x = center_x / (
                        float(self.config["IMAGE_W"]) / self.config["GRID_W"]
                    )
                    center_y = 0.5 * (obj["ymin"] + obj["ymax"])
                    center_y = center_y / (
                        float(self.config["IMAGE_H"]) / self.config["GRID_H"]
                    )

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if (
                        grid_x < self.config["GRID_W"]
                        and grid_y < self.config["GRID_H"]
                    ):
                        obj_indx = self.config["LABELS"].index(obj["name"])

                        center_w = (obj["xmax"] - obj["xmin"]) / (
                            float(self.config["IMAGE_W"]) / self.config["GRID_W"]
                        )  # unit: grid cell
                        center_h = (obj["ymax"] - obj["ymin"]) / (
                            float(self.config["IMAGE_H"]) / self.config["GRID_H"]
                        )  # unit: grid cell

                        box = [center_x, center_y, center_w, center_h]

                        # find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou = -1

                        shifted_box = BoundBox(0, 0, center_w, center_h)

                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou = bbox_iou(shifted_box, anchor)

                            if max_iou < iou:
                                best_anchor = i
                                max_iou = iou

                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4] = 1.0
                        y_batch[
                            instance_count, grid_y, grid_x, best_anchor, 5 + obj_indx
                        ] = 1

                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box

                        true_box_index += 1
                        true_box_index = true_box_index % self.config["TRUE_BOX_BUFFER"]

            # assign input image to x_batch
            if self.norm != None:
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    if obj["xmax"] > obj["xmin"] and obj["ymax"] > obj["ymin"]:
                        cv2.rectangle(
                            img[:, :, ::-1],
                            (obj["xmin"], obj["ymin"]),
                            (obj["xmax"], obj["ymax"]),
                            (255, 0, 0),
                            3,
                        )
                        cv2.putText(
                            img[:, :, ::-1],
                            obj["name"],
                            (obj["xmin"] + 2, obj["ymin"] + 12),
                            0,
                            1.2e-3 * img.shape[0],
                            (0, 255, 0),
                            2,
                        )

                x_batch[instance_count] = img

            # increase instance counter in current batch
            instance_count += 1

        self.counter += 1
        # print ' new batch created', self.counter

        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)
        self.counter = 0

    def aug_image(self, train_instance, jitter):
        image_name = train_instance["filename"]
        image = self.reader.open(str(image_name))
        if image is None:
            raise IOError(f"Error opening image: {image_name}, image is None")
        
        image = image.getUCharPatch(startX=train_instance['min_x'], startY=train_instance['min_y'], width=train_instance['width'], height=train_instance['height'], level=0)
        
        h, w, _ = image.shape
        x_pad = self.width - w
        y_pad = self.height - h

        padding = ((0,y_pad), (0,x_pad), (0,0))
        image = np.pad(image, pad_width = padding, mode = 'constant')

        h, w, _ = image.shape

        all_objs = copy.deepcopy(train_instance["object"])

        if jitter:
            ### scale the image
            scale = np.random.uniform() / 10.0 + 1.0
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

            ### translate the image
            max_offx = (scale - 1.0) * w
            max_offy = (scale - 1.0) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = image[offy : (offy + h), offx : (offx + w)]

            ### flip the image
            flip = np.random.binomial(1, 0.5)
            if flip > 0.5:
                image = cv2.flip(image, 1)

            image = self.aug_pipe.augment_image(image)

        # resize the image to standard size
        image = cv2.resize(image, (self.config["IMAGE_H"], self.config["IMAGE_W"]))
        image = image[:, :, ::-1]

        # fix object's position and size
        for obj in all_objs:
            for attr in ["xmin", "xmax"]:
                if jitter:
                    obj[attr] = int(obj[attr] * scale - offx)

                obj[attr] = int(obj[attr] * float(self.config["IMAGE_W"]) / w)
                obj[attr] = max(min(obj[attr], self.config["IMAGE_W"]), 0)

            for attr in ["ymin", "ymax"]:
                if jitter:
                    obj[attr] = int(obj[attr] * scale - offy)

                obj[attr] = int(obj[attr] * float(self.config["IMAGE_H"]) / h)
                obj[attr] = max(min(obj[attr], self.config["IMAGE_H"]), 0)

            if jitter and flip > 0.5:
                xmin = obj["xmin"]
                obj["xmin"] = self.config["IMAGE_W"] - obj["xmax"]
                obj["xmax"] = self.config["IMAGE_W"] - xmin

        return image, all_objs


def parse_annotation(ann_dir, img_dir, labels=[]):
    reader = mir.MultiResolutionImageReader()
    all_imgs = []
    seen_labels = {}

    # for ann in sorted(os.listdir(ann_dir)):
    for data_file, XML_file in zip(os.listdir(img_dir), os.listdir(ann_dir)): 
        XML_file_name = os.path.join(ann_dir, XML_file)
        data_file_name = os.path.join(img_dir, data_file)
        
        
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

        for index, elem in enumerate(annotations):
            if elem.label.name == "roi":
                roi = {"object": []}
                roi["filename"] = os.path.join(img_dir, data_file_name)
            
                [min_x, min_y, max_x, max_y] = elem.bounds

                # Add coordinates of the roi within the image for slicing
                roi['min_x'] = min_x
                roi['min_y'] = min_y
                roi['max_x'] = max_x
                roi['max_y'] = max_y

                roi["width"] = max_x-min_x
                roi["height"] = max_y-min_y


                for annotation in annotations:
                    obj = {}
                    if annotation.label.name == "lymphocytes and plasma cells" and elem.contains(annotation):
                        obj["name"] = "lymphocytes and plasma cells"

                        if obj["name"] in seen_labels:
                            seen_labels[obj["name"]] += 1
                        else:
                            seen_labels[obj["name"]] = 1

                        if len(labels) > 0 and obj["name"] not in labels:
                            break
                        else:
                            roi["object"] += [obj]

                        box = np.array(annotation.coordinates[:-1])
                        obj["xmin"] = np.amin(box, axis = 0)[0]-min_x
                        obj["xmax"] = np.amax(box, axis = 0)[0]-min_x
                        obj["ymin"] = np.amin(box, axis = 0)[1]-min_y
                        obj["ymax"] = np.amax(box, axis = 0)[1]-min_y

                    if len(roi["object"]) > 0:
                        all_imgs += [roi]

    return all_imgs, seen_labels

def YOLO_network(input_img,true_bxs,CLASS):

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

# Set paths to the data
path = os.getcwd()
parent = os.path.dirname(path)

X_DIR = os.path.join(parent,r'data_sample\wsirois\wsi-level-annotations\images')
y_DIR = os.path.join(parent,r'data_sample\wsirois\wsi-level-annotations\annotations-tissue-cells-xmls')
MSKS_DIR =os.path.join(parent, r'data_sample\wsirois\wsi-level-annotations\annotations-tissue-cells-masks')

all_imgs, seen_labels = parse_annotation(y_DIR, X_DIR, ["lymphocytes and plasma cells"])
max_height = 2048 # max([img['height'] for img in all_imgs])+1
max_width =  2048#max_height#max([img['width'] for img in all_imgs])+1

print(max_height, max_width)

# Set the parameters for the detection of the right lung

LABELS = ["lymphocytes and plasma cells"]

IMAGE_H, IMAGE_W = max_height, max_width
GRID_H,  GRID_W  = 64 , 64
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

def space_to_depth_x2(x):
    return tf.nn.space_to_depth(x, block_size=2)


early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=10, 
                           mode='min', 
                           verbose=1)

checkpoint = ModelCheckpoint('weights_left_right_lung.h5',
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
train_batch = BatchGenerator(all_imgs[:train_valid_split], max_width, max_height, generator_config, norm=utils.normalize)
valid_batch = BatchGenerator(all_imgs[train_valid_split:], max_width, max_height, generator_config, norm=utils.normalize)

# define number of epochs
n_epoch = 2
learning_rate = 1e-5

# Define Adam optimizer
optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.75, epsilon=1e-08, decay=0.0)

dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))

model = YOLO_network(input_image, true_boxes, CLASS)

# compile YOLO model
loss = Loss(16, 64, 64, ANCHORS, n_boxes = 5)
model.compile(loss=loss, optimizer=optimizer)

# do training
#print(train_batch.shape)
model.fit(train_batch,
            validation_data = valid_batch,
            epochs=n_epoch,
            callbacks=[early_stop, checkpoint])