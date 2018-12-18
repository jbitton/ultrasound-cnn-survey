from __future__ import print_function
import argparse
import cv2
import json
import logging
import os
import numpy as np
from datetime import datetime
from shapely.geometry import shape


logging.basicConfig(level=logging.INFO)
IMG_ID = -1
ANN_ID = -1


def img_id_increment():
    global IMG_ID
    IMG_ID += 1
    temp = IMG_ID
    return temp


def img_id_decrement():
    global IMG_ID
    IMG_ID -= 1
    temp = IMG_ID
    return temp


def ann_id_increment():
    global ANN_ID
    ANN_ID += 1
    temp = ANN_ID
    return temp


def get_segmentation(border):
    segmentation = []
    for vertex in border:
        segmentation.append([vertex[0], vertex[1]])
        # segmentation.append(vertex[1])
    return segmentation


def get_bbox(border):
    bottom_border = -1
    top_border = 1000000
    right_border = -1
    left_border = 1000000
    for vertex in border:
        bottom_border = max(bottom_border, vertex[1])
        top_border = min(top_border, vertex[1])
        right_border = max(right_border, vertex[0])
        left_border = min(left_border, vertex[0])
    height = bottom_border - top_border
    width = right_border - left_border
    return [left_border, top_border, width, height]


def create_image_json(img_id, info, case_file, img_format=".jpg"):
    file_name = "{:08d}{}".format(img_id, img_format)
    image_json = {
        "license": 1,
        "file_name": file_name,
        "nerve_file": case_file,
        "height": info["height"],
        "width": info["width"],
        "date_captured": str(datetime.now()),
        "id": img_id,
    }
    return image_json


def create_annotation_json(img_id, case_path, case_file):
    mask_file = case_file.split('.')[0] + '_mask.tif'
    mask = cv2.imread(os.path.join(case_path, mask_file))
    mask_bw = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask_bw = cv2.threshold(mask_bw, 127, 255, 0)[1]

    if np.sum(mask_bw) > 0:
        border = []
        vertices = cv2.findContours(mask_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1][0].tolist()
        for vertex in vertices:
            border.append(vertex[0])
        x, y = zip(*border)
        area = shape({'type': 'Polygon', 'coordinates': [zip(x, y)]}).area
        bbox = get_bbox(border)
        segmentation = [get_segmentation(border)]
        annotation = {
            "id": ann_id_increment(),
            "image_id": img_id,
            "category_id": 1,
            "segmentation": segmentation,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        }
    else:
        annotation = {
            "id": ann_id_increment(),
            "image_id": img_id,
            "category_id": 0,
            "segmentation": [],
            "bbox": [],
            "area": 0,
            "iscrowd": 0
        }
    return annotation


def create_instances_json(images, annotations, data_folder, dataset):
    info = {
        "description": "Ultrasound Nerve Data",
        "url": "https://www.kaggle.com/c/ultrasound-nerve-segmentation",
        "version": "1.0",
        "year": 2018,
        "contributor": "jbitton",
        "date_created": str(datetime.now())
    }
    licenses = [
        {
            "id": 1,
            "name": "Dummy license",
            "url": "dummy_license_url"
        }
    ]
    categories = [
        {
            'id': 1,
            'name': 'nerve',
            'supercategory': 'body_part',
        },
    ]
    instances_json = {
        "info": info,
        "licenses": licenses,
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    json_file = os.path.join(data_folder, "instances_{}.json".format(dataset))
    with open(json_file, 'w') as fp:
        json.dump(instances_json, fp)


def read_case(case_path, data_folder, case_file):
    image = cv2.imread(os.path.join(case_path, case_file))
    img_id = img_id_increment()
    output_file = "{:08d}{}".format(img_id, '.jpg')
    cv2.imwrite(os.path.join(data_folder, output_file), image)

    info = {
        'id': img_id,
        'height': image.shape[0],
        'width': image.shape[1]
    }
    image_json = create_image_json(img_id, info, case_file, img_format=".jpg")
    annotation = create_annotation_json(img_id, case_path, case_file)

    return image_json, annotation


def validate_folders(cases_folder, data_folder):
    if not os.path.exists(cases_folder):
        raise (RuntimeError(f"Could not find {cases_folder}, please download the data from Kaggle"))

    if not os.path.exists(data_folder):
        logging.warning(f"Could not find {data_folder}. Creating new folder and processing data")
        os.mkdir(data_folder)

    if not os.path.exists(os.path.join(data_folder, "annotations")):
        os.mkdir(os.path.join(data_folder, "annotations"))

    if not os.path.exists(os.path.join(data_folder, "train")):
        os.mkdir(os.path.join(data_folder, "train"))

    if not os.path.exists(os.path.join(data_folder, "val")):
        os.mkdir(os.path.join(data_folder, "val"))


def main():
    parser = argparse.ArgumentParser(description='Processing Nerve Data to COCO format')
    parser.add_argument('--data', type=str, default='nerve_data', metavar='d', help="downloaded data directory")
    parser.add_argument('--save', type=str, default='nerve_coco_data', metavar='s', help="directory where data saved.")
    parser.add_argument('--split', type=int, default=10, metavar='S', help="# to split b/t train/val. def: 10")
    parser.add_argument('--enable-log', type=str, default='y', metavar='L', help="flag to enable logger")
    args = parser.parse_args()

    root_dir = os.getcwd()
    cases_folder = os.path.join(root_dir, args.data)
    data_folder = os.path.join(root_dir, args.save)
    data_split = args.split
    if args.enable_log == "n":
        logging.disable(level=logging.INFO)

    images_train = []
    annotations_train = []
    images_val = []
    annotations_val = []
    case_numbers = set()

    validate_folders(cases_folder, data_folder)
    for file in os.listdir(cases_folder):
        if '_mask.tif' not in file and '.tif' in file:
            case_number = int(file.split('_')[0])
            if case_number not in case_numbers:
                case_numbers.add(case_number)
            if case_number % data_split == 0:  # make this a validation case
                img, ann = read_case(cases_folder, os.path.join(data_folder, "val"), file)
                images_val.append(img)
                annotations_val.append(ann)
            else:
                img, ann = read_case(cases_folder, os.path.join(data_folder, "train"), file)
                images_train.append(img)
                annotations_train.append(ann)

    create_instances_json(images_train, annotations_train, os.path.join(data_folder, "annotations"), "train")
    create_instances_json(images_val, annotations_val, os.path.join(data_folder, "annotations"), "val")


if __name__ == "__main__":
    main()

