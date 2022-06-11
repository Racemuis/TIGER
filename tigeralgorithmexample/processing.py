from pathlib import Path
from typing import List
from PIL import Image

from keras import backend as K

K.clear_session()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import numpy as np
from tqdm import tqdm
import tensorflow as tf



from .gcio import (
    TMP_DETECTION_OUTPUT_PATH,
    TMP_SEGMENTATION_OUTPUT_PATH,
    TMP_TILS_SCORE_PATH,
    copy_data_to_output_folders,
    get_image_path_from_input_folder,
    get_tissue_mask_path_from_input_folder,
    initialize_output_folders,
)

from .rw import (
    READING_LEVEL,
    WRITING_TILE_SIZE,
    DetectionWriter,
    SegmentationWriter,
    TilsScoreWriter,
    open_multiresolutionimage_image,
)

from . import utils
from .detect import run

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
                    
                    box = utils.BoundingBox(x, y, w, h, confidence, classes)
                    
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

def pad_ensure_division(h, w, division):

    def compute_pad(s, d):
        if s % d != 0:
            p = 0
            while True:
                if (s + p) % d == 0:
                    return p
                p += 1
        return 0

    py = compute_pad(h, division)
    px = compute_pad(w, division)
    padding = (0, py), (0, px)
    return padding

def process_image_tile_to_segmentation(model,
    image_tile: np.ndarray, tissue_mask_tile: np.ndarray
) -> np.ndarray:

    """Segment the image tile using UNet.
    

    Args:
        image_tile (np.ndarray): [description]
        tissue_mask_tile (np.ndarray): [description]

    Returns:
        np.ndarray: [description]

    Returns of Unet: 
    Return Label 0 contains: (3, 4, 5, 7)
    Return Label 1 contains: (1)
    Return Label 2 contains: (2, 6)
 
        1: Invasive tumor
        2: Tumor-associated stroma
        3: In-situ tumor
        4: Healthy glands
        5: Necrosis not in-situ 
        6: Inflamed stroma
        7: Rest

    """

    # Pad image if the size is not divisable by total downsampling rate (8) in UNet 
    h, w, _ = image_tile.shape
    (py0, py1), (px0, px1) = pad_ensure_division(h, w, 8)
    padding = ((py0, py1), (px0, px1), (0, 0))
    pad_imgs = np.pad(image_tile, pad_width=padding, mode='constant')
    
    # Run Unet model
    output = np.argmax(model.predict(np.expand_dims(pad_imgs,0), batch_size=1), axis = -1)

    # Remove the padding
    output = output[0, py0:py0+h, px0:px0+w]

    return output * tissue_mask_tile


def process_image_tile_to_detections(
    model,
    image_tile: np.ndarray,
    segmentation_mask: np.ndarray,
) -> List[tuple]:
    """ Detect Lymphocytes using YOLOv5. 
    Retrieved from: https://github.com/ultralytics/yolov5

    Args:
        image_tile (np.ndarray): [description]
        tissue_mask_tile (np.ndarray): [description]

    Returns:
        List[tuple]: list of tuples (x,y) coordinates of detections
    """

    # Save the tile as an image
    im = Image.fromarray(image_tile)
    path = r"./image_tile.png"
    im.save(path)


    # Get output from YOLOv5 (box sizes are normalized to the image size)
    boxes = run(weights= model, source = path, conf_thres =0.1, imgsz = (image_tile[0], image_tile[1]))

    xs = []
    ys = []
    probabilities = []
    for box in boxes:
        # Denormalize the boxes
        xs.append(box[0])*image_tile[0]
        ys.append(box[1])*image_tile[1]
        probabilities.append(box[2])

    return list(zip(xs, ys, probabilities))


def process_segmentation_detection_to_tils_score(
    segmentation_path: Path, detections: List[tuple]
) -> int:
    """Example function that shows processing a segmentation mask and corresponding detection for the computation of a tls score.
    
    NOTE 
        This code is only made for illustration and is not meant to be taken as valid processing step.

    Args:
        segmentation_mask (np.ndarray): [description]
        detections (List[tuple]): [description]

    Returns:
        int: til score (between 0, 100)
    """

    level = 4
    cell_area_level_1 = 16*16

    image = open_multiresolutionimage_image(path=segmentation_path)
    width, height = image.getDimensions()
    slide_at_level_4 = image.getUCharPatch(0, 0, int(width / 2**level), int(height / 2**level), level)
    area = len(np.where(slide_at_level_4 == 2)[0])
    cell_area = (cell_area_level_1//2**4)
    n_detections = len(detections)
    if cell_area == 0 or n_detections == 0:
        return 0
    value = min(100, int(area / (n_detections / cell_area)))
    return value


def process():
    """Proceses a test slide"""

    level = READING_LEVEL
    tile_size = WRITING_TILE_SIZE # should be a power of 2

    initialize_output_folders()

    # get input paths
    image_path = get_image_path_from_input_folder()
    tissue_mask_path = get_tissue_mask_path_from_input_folder()

    print(f'Processing image: {image_path}')
    print(f'Processing with mask: {tissue_mask_path}')

    # open images
    image = open_multiresolutionimage_image(path=image_path)
    tissue_mask = open_multiresolutionimage_image(path=tissue_mask_path)

    # get image info
    dimensions = image.getDimensions()
    spacing = image.getSpacing()

    # create writers
    print(f"Setting up writers")
    segmentation_writer = SegmentationWriter(
        TMP_SEGMENTATION_OUTPUT_PATH,
        tile_size=tile_size,
        dimensions=dimensions,
        spacing=spacing,
    )
    detection_writer = DetectionWriter(TMP_DETECTION_OUTPUT_PATH)
    tils_score_writer = TilsScoreWriter(TMP_TILS_SCORE_PATH)

    # Load the models
    YOLO = tf.keras.models.load_model('./Final_Models/best.pt')
    UNet = tf.keras.models.load_model('./Final_Models/model_Unet_e3_all_data.h5')

    print("Processing image...")
    # loop over image and get tiles
    for y in tqdm(range(0, dimensions[1], tile_size)):
        for x in range(0, dimensions[0], tile_size):
            tissue_mask_tile = tissue_mask.getUCharPatch(
                startX=x, startY=y, width=tile_size, height=tile_size, level=level
            ).squeeze()
            
            if not np.any(tissue_mask_tile):
                continue

            image_tile = image.getUCharPatch(
                startX=x, startY=y, width=tile_size, height=tile_size, level=level
            )

            # segmentation
            segmentation_mask = process_image_tile_to_segmentation(model = UNet,
                image_tile=image_tile, tissue_mask_tile=tissue_mask_tile
            )

            segmentation_writer.write_segmentation(tile=segmentation_mask, x=x, y=y)
# detection
            detections = process_image_tile_to_detections(model = YOLO, 
                image_tile=image_tile, segmentation_mask=segmentation_mask
            )
            
            detection_writer.write_detections(
                detections=detections, spacing=spacing, x_offset=x, y_offset=y
            )

    print("Saving...")
    # save segmentation and detection
    segmentation_writer.save()
    detection_writer.save()

    print('Number of detections', len(detection_writer.detections))
    
    print("Compute tils score...")
    # compute tils score
    tils_score = process_segmentation_detection_to_tils_score(
        TMP_SEGMENTATION_OUTPUT_PATH, detection_writer.detections
    )
    tils_score_writer.set_tils_score(tils_score=tils_score)

    print("Saving...")
    # save tils score
    tils_score_writer.save()

    print("Copy data...")
    # save data to output folder
    copy_data_to_output_folders()

    print("Completed!")
