import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from BoundBox import tf_iou

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
