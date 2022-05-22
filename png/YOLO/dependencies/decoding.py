import tensorflow as tf
import numpy as np

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

