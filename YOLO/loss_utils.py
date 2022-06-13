import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as pltpatches

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


# decoding function, only for one class,
# decoding function, only for one prediction
# decoding function, hard coded anchors
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