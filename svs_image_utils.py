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
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

TILE_SIZE = 5000
OUTPUT_SIZE = 256
OUTPUT_DIR = 'revamp_tile_output'
INPUT_DIR = '/mnt/e/CATCH/PKG - CATCH'
TEST_IMAGE = '/mnt/e/CATCH/PKG - CATCH/MCT/MCT_33_1.svs'
COCO_PATH = '/mnt/e/CATCH/PKG - CATCH/CATCH.json'

def process_one_image(args : tuple):
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
    img, anns, output_dir, local_img_id, local_ann_id, test = args

    #slice the image

    #open the slide

    slide = openslide.OpenSlide(img)

    #set slide width and height, then get number of images
    slide_width, slide_height = slide.dimensions
    num_vertical = slide_height // TILE_SIZE + 1
    num_horizontal = slide_width // TILE_SIZE + 1
    print(f'my image ID: {local_img_id}, my number of tiles: {num_vertical * num_horizontal}')
    scale = (OUTPUT_SIZE - 1)/TILE_SIZE

    #print(f'slide dimensions: {slide.dimensions}\nnum_vertical: {num_vertical}\nnum_horizontal: {num_horizontal}')
    output_images = []
    output_annotations = []
    #iterate over the number of vertical and horizontal slides
    for v in range(num_vertical):
        for h in range(num_horizontal):
            tile_x = h * TILE_SIZE
            tile_y = v * TILE_SIZE
            #print(f"Currently on slide{v}, {h}")

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
            

            #figure out the tile intersection
            tile_polygon = shapely.box(tile_x, tile_y, tile_xf, tile_yf)

            intersections = []

            for ann in anns:
                #if there is something in the segmentation section (remember there is no crowd annotations)
                if ann['segmentation']:
                    poly_coords = np.array(ann['segmentation']).reshape(-1, 2)
                    poly = shapely.Polygon(poly_coords)
                    if not poly.is_valid:
                        poly = shapely.make_valid(poly)
                    
                    intersection = tile_polygon.intersection(poly)

                    #if there's no intersection, move along
                    if intersection.is_empty:
                        continue

                    intersections.append({'cat_id' : ann['category_id'], 'intersection' : intersection})
                    

            for dict in intersections:
                intersection = dict['intersection']

                for otherdict in intersections:
                    if intersection.equals(otherdict['intersection']):
                        continue
                    elif intersection.covers(otherdict['intersection']):
                        intersection = intersection.difference(otherdict['intersection'])
                
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
                    "category_id": dict['cat_id'],
                    "segmentation": scaled_coords,
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "iscrowd": 0,
                    "area": intersection.area * scale * scale
                })
                local_ann_id += 1
                #for tomorrow, figure out why the annotations aren't properly figuring out the scaled coordinates
                #specifically, why we're not getting as many annotations as we're getting masks?
                if test:
                    mask = generate_mask(scaled_coords)
                    Image.fromarray(mask).save(os.path.join(output_dir, f"{filename.split('.')[0]}_ann{local_ann_id - 1}.png"))
            local_img_id += 1

    return output_images, output_annotations

def generate_mask(segmentation : list, cat_id : int = 255):
    mask = np.zeros((256, 256), dtype = np.uint8)
    for seg in segmentation:
        poly_coords = np.array(seg).reshape(-1, 2)
        rr, cc = polygon(poly_coords[:, 0], poly_coords[:, 1])
        mask[cc, rr] ^= cat_id
    return mask

def test():
    coco = COCO(COCO_PATH)
    anns = coco.loadAnns(coco.getAnnIds(3))
    output_path = os.path.join(OUTPUT_DIR, 'test')
    os.makedirs(output_path, exist_ok = True)
    args = [TEST_IMAGE, anns, output_path, 1, 1, True]

    images, annotations, img_id, ann_id = process_one_image(args)
    output_json = {
        "images": images,
        "annotations": annotations,
        "categories": coco.loadCats(coco.getCatIds())
    }

    out_path = os.path.join(OUTPUT_DIR, "test", "annotations", "test_annotations.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_json, f)

def get_tumor_from_file(filename : str):
    return filename.split('_')[0]

def create_output_dirs(output_dir : str, tumor_type : str):
    dir = os.path.join(output_dir, tumor_type, 'images')
    os.makedirs(dir)
    return dir

def main(test = False):
    coco = COCO(COCO_PATH)
    img_ids = coco.getImgIds()
    images = coco.loadImgs(img_ids)
    if test:
        images = images[:2]
    output_images = []
    output_annotations = []
    img_id = 1
    ann_id = 1
    args = []

    for image in images:
        tumor = get_tumor_from_file(image['file_name'])
        output_path = os.path.join(OUTPUT_DIR, tumor)
        os.makedirs(output_path, exist_ok = True)
        print(f"Made directory: {output_path}")
        image_filename = os.path.join(INPUT_DIR, tumor, image['file_name'])
        anns = coco.loadAnns(coco.getAnnIds(image['id']))
        args.append((image_filename, anns, output_path, img_id, ann_id, test))
        img_id += (image['height'] // 5000 + 1) * (image['width'] // 5000 + 1)
        ann_id += 10000

    print(f'Using {min(8, len(args))} threads.')
    with Pool(processes=min(8, len(args))) as pool:
        results = list(tqdm(pool.imap(process_one_image, args), total=len(args)))

    all_images = []
    all_annotations = []
    for img_list, ann_list in results:
        all_images.extend(img_list)
        all_annotations.extend(ann_list)

    output_json = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": coco.loadCats(coco.getCatIds())
    }

    ann_path = os.path.join(OUTPUT_DIR, "annotations", "revamp_tile_coco.json")
    os.makedirs(os.path.dirname(ann_path), exist_ok=True)
    with open(ann_path, "w") as f:
        json.dump(output_json, f)

    print(f'Saved {len(all_images)} tiles and {len(all_annotations)} annotations')

if __name__ == '__main__':
    main(test = False)