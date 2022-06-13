import json
import os
from tqdm import tqdm

ANNOTATION_DIR = r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data\wsirois\roi-level-annotations\tissue-cells\masks_v5'
IMAGE_DIMENSION = 1024

with open(r'C:\Users\Racemuis\Documents\intelligent systems in medical imaging\project\data_sample\wsirois\roi-level-annotations\tissue-cells\tiger-coco.json', 'r') as file:
  data = json.load(file)


image_ids = [img['id'] for img in data['images']]

cls = 0

for id in tqdm(image_ids, desc="Parsing annotations"):
    bbox_present = False
    
    for a in data['annotations']:
        if a['image_id'] == id:
            bbox_present = True
            bbox = a['bbox']
            x_top_left = bbox[0]/IMAGE_DIMENSION
            y_top_left = bbox[1]/IMAGE_DIMENSION
            width = bbox[2]/IMAGE_DIMENSION
            height = bbox[3]/IMAGE_DIMENSION

            x_centre = x_top_left + width/2
            y_centre = y_top_left + height/2

            with open(os.path.join(ANNOTATION_DIR, f'{id}.txt'), 'a') as f:
                f.write(f"{cls} {x_centre} {y_centre} {width} {height}\n")
    
    if not bbox_present:
        with open(os.path.join(ANNOTATION_DIR, f'{id}.txt'), 'a') as f:
            f.write(f"")


