import os
import json
import random
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict


base_dir = './tiles_output/tiles_output/'  
images_dir = os.path.join(base_dir, 'images')
annotations_path = os.path.join(base_dir, 'annotations', 'test_annotations.json')
output_dir = os.path.join(base_dir, 'yolo_dataset')


train_ratio = 0.8

random.seed(42)

for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

with open(annotations_path, 'r') as f:
    coco = json.load(f)

categories = coco['categories']
cat_id_to_yolo = {cat['id']: idx for idx, cat in enumerate(categories)}

ann_by_img = defaultdict(list)
for ann in coco['annotations']:
    ann_by_img[ann['image_id']].append(ann)

images = {img['id']: img for img in coco['images']}

# Splig
img_ids = list(images.keys())
train_ids, val_ids = train_test_split(img_ids, train_size=train_ratio, random_state=42)

# Helper function to process images
def process_image(img_id, split):
    img_info = images[img_id]
    img_filename = img_info['file_name']
    img_w = img_info['width']
    img_h = img_info['height']

    src_img_path = os.path.join(images_dir, img_filename)
    
    if not os.path.exists(src_img_path):
        print(f"Warning: Image {img_filename} not found at {src_img_path}!")
        return
    
    dst_img_path = os.path.join(output_dir, f'images/{split}', os.path.basename(img_filename))
    
    # use chutil to copy image, not sure how this works very well
    shutil.copy2(src_img_path, dst_img_path)

    # label
    label_filename = os.path.splitext(os.path.basename(img_filename))[0] + '.txt'
    label_path = os.path.join(output_dir, f'labels/{split}', label_filename)

    with open(label_path, 'w') as f:
        anns = ann_by_img.get(img_id, [])
        for ann in anns:
            bbox = ann['bbox']
            x_min, y_min, width, height = bbox
            x_center = (x_min + width/2) / img_w
            y_center = (y_min + height/2) / img_h
            width /= img_w
            height /= img_h
            category_id = ann['category_id']
            yolo_class = cat_id_to_yolo[category_id]
            f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# training imgs
print("Processing training images...")
for img_id in tqdm(train_ids):
    process_image(img_id, 'train')

# test imgs
print("Processing validation images...")
for img_id in tqdm(val_ids):
    process_image(img_id, 'val')

# create data.yaml, needed to train model
yaml_path = os.path.join(output_dir, 'data.yaml')
class_names = [cat['name'] for cat in categories]

with open(yaml_path, 'w') as f:
    f.write(f"path: {output_dir}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")
    f.write(f"names: {class_names}\n")
