import numpy as np


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



