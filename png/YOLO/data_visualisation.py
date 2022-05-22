import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatches

from PIL import Image
from dependencies import BatchGenerator

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
                    obj["width"] = box[2]
                    obj["height"] = box[3]

            #if len(img["object"]) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels


def plot_image_with_boxes(image, boxes):
    f, ax = plt.subplots()
    ax.imshow(image)
    # draw boundingboxes
    for box in boxes:
        # Create a Rectangle patch
        x = box['xmin']
        y = box['ymin']
        w = box['width']
        h = box['height']
        rect = pltpatches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()


# Set paths to the data
path = os.getcwd()
parent = os.path.dirname(path)

X_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsirois\roi-level-annotations\tissue-cells\images'
y_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsirois\roi-level-annotations\tissue-cells\tiger-coco.json'
all_imgs, seen_labels = parse_annotation(y_DIR, X_DIR, ["lymphocytes and plasma cells"])

batch = BatchGenerator(all_imgs[0], all_imgs[0]['width'], all_imgs[0]['height'], None, norm=None)


boxes = all_imgs[0]['object']
img = Image.open(str(all_imgs[0]['filename'])).convert("RGB")
pixels = np.asarray(img)
plot_image_with_boxes(pixels, boxes)