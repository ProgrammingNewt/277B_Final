import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import cv2
import os
import numpy as np

#category colors for each coco category

category_colors = {
    1: "#d1e0e0",  #bone - light grey
    2: "#a3b9b9",  #cartilage - gray blue
    3: "#f2c9c9",  #dermis - light pink
    4: "#f0e6d2",  #epidermis - beige
    5: "#e0d7b3",  #subcutis - olive
    6: "#f9d6b6",  #inflamm/necrosis - light orange
    7: "#e74c3c",  #melanoma - red
    8: "#8e44ad",  #plasmacytoma - purple
    9: "#c0392b",  #mast cell tumor - dark red
    10: "#f39c12", #pnst - yellow
    11: "#e67e22", #scc - orange
    12: "#d35400", #trichoblastoma - dark orange
    13: "#27ae60"  #histiocytoma - green
}



def show_image_with_segmentation(coco_json_path, image_folder, image_id=1):
    """
    Plots an image with its segmentation mask.

    Inputs
    ------
    coco_json_path : str
        put the path to the coco json here. Should be something like ".../tiles_output/annotations/test_annotations.json"
    
    image_folder : str
        put the path to the image folder here. Should be something like ".../tiles_output"
    
    image_index : int
        the ID of the image you're looking for
    
    Returns
    -------
    None
    """

    #load the coco json and find the image
    coco = COCO(coco_json_path)
    img_info = next((img for img in coco.dataset['images'] if img['id'] == image_id), None)
    file_name = img_info['file_name']
    print(file_name)
    img_path = os.path.join(image_folder, 'images', file_name)

    #load the image using cv2
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #find the ids of the annotations
    ann_ids = coco.getAnnIds(imgIds=[image_id])
    anns = coco.loadAnns(ann_ids)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    ax = plt.gca()

    for ann in anns:
        if 'segmentation' in ann:
            #for each segmentation in the annotations
            for seg in ann['segmentation']:
                #reshape the segmentation polygon into [[x, y], [x1, y1], ...]
                poly = np.array(seg).reshape((len(seg)//2, 2))

                #make a polygon using patches library
                patch = patches.Polygon(poly, closed = True, facecolor=category_colors[ann['category_id']], edgecolor='black', linewidth=2, alpha=0.5)
                ax.add_patch(patch)
                

    plt.title(f"Image: {file_name} with {len(anns)} annotations")
    plt.axis("off")
    plt.show()

def get_images_with_segments(coco_json_path, image_folder, num_segments):
    """
    gets image ids with number of annotations equal to num_segments

    Inputs
    ------
    coco_json_path : str
        put the path to the coco json here. Should be something like ".../tiles_output/annotations/test_annotations.json"
    
    image_folder : str
        put the path to the image folder here. Should be something like ".../tiles_output"
    
    image_index : int
        the ID of the image you're looking for
    
    Returns
    -------
    list(int)
        list of valid image ids
    """
    coco = COCO(coco_json_path)
    valid_img_ids = []

    for img_id in coco.imgs:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        if len(ann_ids) == num_segments:
            valid_img_ids.append(img_id)
    if not valid_img_ids:
        raise ValueError('No images found with that number of segments.')
    return valid_img_ids

if __name__ == '__main__':
    coco_path = "../tiles_output/annotations/test_annotations.json"
    img_path = "../tiles_output"
    valid_ids = get_images_with_segments(coco_path, img_path, 3)
    test_id = np.random.choice(valid_ids)
    show_image_with_segmentation(coco_path, img_path, test_id)