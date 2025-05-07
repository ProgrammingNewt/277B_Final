"""
Pseudocode:

process one image:
    slice into 5000 x 5000 parts

    record the original location of the part

    resize image to 600 x 600
    if it's a corner or edge piece:
        add padding, then resize

    for annotation in image:
        if the tile intersects with the annotation:
            clip the annotation to the tile
            resize the bounding box of the annotation

main function
    load imgs
    load anns
    for img in imgs:
        process one image
            should save imgs, return anns
        add anns to dict of anns
"""
import openslide
import os
import cv2
from pycocotools.coco import COCO
from PIL import Image
import shapely
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
import json

TILE_SIZE = 5000
OUTPUT_SIZE = 256
OUTPUT_DIR = 'revamp_tile_output'
TEST_IMAGE = '/mnt/e/CATCH/PKG - CATCH/MCT/MCT_01_1.svs'
COCO_PATH = '/mnt/e/CATCH/PKG - CATCH/CATCH.json'

def process_one_image(img : str, anns : list, output_dir : str, local_img_id : int, local_ann_id : int, test = False):
    """
    img : str
        filepath of image
    anns : list
        list of annotation dicts from COCO
    output_dir : str
        output directory of the file
    starting_img_id : int
        the img id we're starting with
    starting_ann_id : int
        the ann id we're starting with

    returns
    -------
    list
        list of image dicts given as: id, file_name, width, height
    
    list
        list of anns given as: id, image_id, category_id, segmentation, bbox, iscrowd, area
    """
    #slice the image

    #open the slide
    slide = openslide.OpenSlide(img)

    #set slide width and height, then get number of images
    slide_width, slide_height = slide.dimensions
    num_vertical = slide_height // TILE_SIZE + 1
    num_horizontal = slide_width // TILE_SIZE + 1

    scale = (OUTPUT_SIZE - 1)/TILE_SIZE

    print(f'slide dimensions: {slide.dimensions}\nnum_vertical: {num_vertical}\nnum_horizontal: {num_horizontal}')
    output_images = []
    output_annotations = []
    #iterate over the number of vertical and horizontal slides
    for v in range(num_vertical):
        for h in range(num_horizontal):
            tile_x = h * TILE_SIZE
            tile_y = v * TILE_SIZE
            print(f"Currently on slide{v}, {h}")

            #if both out of range
            if(tile_x + TILE_SIZE > slide_width) and (tile_y + TILE_SIZE > slide_height):
                tile = slide.read_region(location = (tile_x, tile_y), level = 0, size = (slide_width - tile_x, slide_height - tile_y))
                tile_xf = slide_width
                tile_yf = slide_height
                #padding
                white_img = Image.new(tile.mode, [TILE_SIZE, TILE_SIZE], color = (255, 255, 255))
                white_img.paste(tile)
                tile = white_img
            
            #if horizontal out of range
            elif(tile_x + TILE_SIZE > slide_width):
                tile = slide.read_region(location = (tile_x, tile_y), level = 0, size = (slide_width - tile_x, TILE_SIZE))
                tile_xf = slide_width
                tile_yf = tile_y + TILE_SIZE
                #padding
                white_img = Image.new(tile.mode, [TILE_SIZE, TILE_SIZE], color = (255, 255, 255))
                white_img.paste(tile)
                tile = white_img

            #if vertical out of  range
            elif(tile_y + TILE_SIZE > slide_height):
                tile = slide.read_region(location = (tile_x, tile_y), level = 0, size = (TILE_SIZE, slide_height - tile_y))
                tile_xf = tile_x + TILE_SIZE
                tile_yf = slide_height
                #padding
                white_img = Image.new(tile.mode, [TILE_SIZE, TILE_SIZE], color = (255, 255, 255))
                white_img.paste(tile)
                tile = white_img
            #otherwise, read the whole thing
            else:
                tile = slide.read_region(location = (tile_x, tile_y), level = 0, size = (TILE_SIZE, TILE_SIZE))
                tile_xf = tile_x + TILE_SIZE
                tile_yf = tile_y + TILE_SIZE

            #save the image
            tile = tile.resize([OUTPUT_SIZE, OUTPUT_SIZE])
            filename = f"{img.split('/')[-1].split('.')[0]}_({v},{h}).png"
            tile.save(os.path.join(output_dir, filename), 'png')

            #make the image dict
            output_images.append({
                'id' : local_img_id,
                'file_name' : filename,
                'width' : 256,
                'height' : 256
            })
            local_img_id += 1

            #figure out the tile intersection
            tile_polygon = shapely.box(tile_x, tile_y, tile_xf, tile_yf)

            for ann in anns:
                #if there is something in the segmentation section (remember there is no crowd annotations)
                if ann['segmentation']:
                    poly_coords = np.array(ann['segmentation']).reshape(-1, 2)
                    poly = shapely.Polygon(poly_coords)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    
                    intersection = tile_polygon.intersection(poly)

                    #if there's no intersection, move along
                    if intersection.is_empty:
                        continue

                    scaled_coords = []
                    if intersection.geom_type == 'Polygon':
                        exterior = np.array(intersection.exterior.coords)
                        exterior = exterior - [tile_x, tile_y]
                        scaled_exterior = exterior * scale
                        scaled_coords.append(scaled_exterior.flatten().tolist())

                        for interior in intersection.interiors:
                            hole = np.array(interior.coords)
                            hole = (hole - [tile_x, tile_y]) * scale
                            scaled_coords.append(hole.flatten().tolist())
                            #if len(hole) > 0:
                                #print(hole)

                    elif intersection.geom_type == 'MultiPolygon':
                        for sub_poly in intersection.geoms:
                            exterior = np.array(sub_poly.exterior.coords)
                            exterior = exterior - [tile_x, tile_y]
                            scaled_exterior = exterior * scale
                            scaled_coords.append(scaled_exterior.flatten().tolist())

                            for interior in sub_poly.interiors:
                                hole = np.array(interior.coords)
                                hole = (hole - [tile_x, tile_y]) * scale
                                scaled_coords.append(hole.flatten().tolist())
                                #if len(hole) > 0:
                                    #print(hole)

                    x_min, y_min, x_max, y_max = intersection.bounds

                    x_min -= tile_x
                    y_min -= tile_y
                    x_max -= tile_x
                    y_max -= tile_y

                    x_min *= scale
                    y_min *= scale
                    x_max *= scale
                    y_max *= scale

                    output_annotations.append({
                        "id": local_ann_id,
                        "image_id": local_img_id,
                        "category_id": ann["category_id"],
                        "segmentation": scaled_coords,
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "iscrowd": 0,
                        "area": intersection.area * scale * scale
                    })
                    local_ann_id += 1

                    if test:
                        mask = generate_mask(scaled_coords)
                        Image.fromarray(mask).save(os.path.join(output_dir, f"{filename.split('.')[0]}_ann{local_ann_id}.png"))

    return output_images, output_annotations, local_img_id, local_ann_id

def generate_mask(segs):
    #print(seg)
    mask = np.zeros((256, 256), dtype = np.uint8)
    #print(segs)
    for seg in segs:
        poly_coords = np.array(seg).reshape(-1, 2)
        rr, cc = polygon(poly_coords[:, 0], poly_coords[:, 1])
        mask[cc, rr] ^= 255
    return mask

def test():
    coco = COCO(COCO_PATH)
    anns = coco.loadAnns(coco.getAnnIds(30))
    output_path = os.path.join(OUTPUT_DIR, 'test')
    os.makedirs(output_path, exist_ok = True)
    images, annotations, img_id, ann_id = process_one_image(TEST_IMAGE, anns, output_path, 0, 0, test = True)
    output_json = {
        "images": images,
        "annotations": annotations,
        "categories": coco.loadCats(coco.getCatIds())
    }

    out_path = os.path.join(OUTPUT_DIR, "test", "annotations", "test_annotations.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_json, f)

if __name__ == '__main__':
    test()