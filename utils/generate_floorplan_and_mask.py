"""
@Author : Sujith Umapathy

This is a simple code to generate floor plan using the opencv library

Task:

1. Use OpenCV to generate rectangle to depict walls
2. Use OpenCV to generate squares to depict objects like bed, wardrobe and others

These images will be used a electronically generated floor plan images

Mask Annotation :

Regions of rectangles & squares - Non Movable Region - Pixel value 255
Remaining Regions - Movable regions - Pixel value 0
"""

import cv2
from configuration import Config as conf
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class WallType(Enum):
    HORIZONTAL = 0
    VERTICAL = 1
    L_TYPE = 2
    BLOCK = 3


def save_co_ordinates(wall_type, value, di):
    li = di.get(wall_type)
    if li is None:
        li = []
    li.append(value)
    di[wall_type] = li


def generate_horizontal_wall(base, start, end, leng, di):
    start_pt = (start, end)
    end_x = start + leng
    end_y = end + conf.width_min_limit
    end_pt = (end_x, end_y)
    cv2.rectangle(base, start_pt, end_pt, 255, -1)
    val = (start_pt, end_pt)
    save_co_ordinates(wall_type=WallType.HORIZONTAL, value=val, di=di)


def generate_vertical_wall(base, start, end, leng, di):
    start_pt = (start, end)
    end_x = start + conf.width_min_limit
    end_y = end + leng
    end_pt = (end_x, end_y)
    cv2.rectangle(base, start_pt, end_pt, 255, -1)
    val = (start_pt, end_pt)
    save_co_ordinates(wall_type=WallType.VERTICAL, value=val, di=di)


def generate_l_wall(base, start, end, leng, di):
    length = np.random.randint(conf.width_min_limit, leng, 1)[0]
    generate_horizontal_wall(base, start, end, length, di=di)
    length = np.random.randint(50, leng + 20, 1)[0]
    generate_vertical_wall(base, start, end, length, di=di)


def generate_wall(types=None, num=3, image=None, dictionary=None):
    if types is not None and image is not None:
        if types.value == 0:
            # horizontal wall
            for i, _ in enumerate(range(num)):
                x = 0
                y = 0
                length = 0

                if i == 0:
                    # generate at the start
                    x = 0
                    y = np.random.randint(conf.width_min_limit, conf.floor_plan_height, 1)[0]
                    length = np.random.randint(conf.width_min_limit, conf.width_max_limit, 1)[0]
                elif i == 1:
                    # generate at the end
                    y = np.random.randint(conf.width_min_limit, conf.floor_plan_height, 1)[0]
                    length = np.random.randint(conf.width_min_limit, conf.width_max_limit, 1)[0]
                    x = conf.floor_plan_width - length
                else:
                    # generate random
                    x = np.random.randint(200, conf.floor_plan_width, 1)[0]
                    y = np.random.randint(200, conf.floor_plan_height, 1)[0]
                    length = np.random.randint(conf.width_min_limit, conf.width_max_limit, 1)[0]
                generate_horizontal_wall(base=image, start=x, end=y, leng=length, di=dictionary)

        elif types.value == 1:
            # vertical wall
            for i, _ in enumerate(range(num)):
                x = 0
                y = 0
                length = 0
                if i == 0:
                    # generate at the start
                    x = np.random.randint(10, conf.floor_plan_height, 1)[0]
                    y = 0
                    length = np.random.randint(conf.width_min_limit, conf.width_max_limit, 1)[0]
                else:
                    # generate at the end
                    x = np.random.randint(10, conf.floor_plan_width, 1)[0]
                    length = np.random.randint(conf.width_min_limit, conf.width_max_limit, 1)[0]
                    y = conf.floor_plan_height - length

                generate_vertical_wall(base=image, start=x - 10, end=y, leng=length, di=dictionary)

        elif types.value == 2:
            # l-type wall
            for _ in range(num):
                x = np.random.randint(10, conf.floor_plan_width, 1)[0]
                y = np.random.randint(10, conf.floor_plan_height, 1)[0]
                length = np.random.randint(50, conf.width_max_limit / 2, 1)[0]
                generate_l_wall(base=image, start=x, end=y, leng=length, di=dictionary)


def generate_floor_plan(base_folder,
                        mask_folder,
                        width=conf.floor_plan_width,
                        height=conf.floor_plan_height,
                        num=400):
    for i in range(0, num):
        mask_image = np.zeros((height, width))
        base_image = np.zeros((height, width, 3))
        base_image.fill(255)
        # gives co-ordinates of all the object types in the electronic floor plan
        semantic_dictionary = {}

        print(f'Working on image {i}')
        # generates mask
        generate_wall(types=WallType.HORIZONTAL, num=3, image=mask_image, dictionary=semantic_dictionary)
        generate_wall(types=WallType.VERTICAL, num=2, image=mask_image, dictionary=semantic_dictionary)
        generate_wall(types=WallType.L_TYPE, num=2, image=mask_image, dictionary=semantic_dictionary)

        # color mask and covert it to spoof original image
        horizontal_wall_list = semantic_dictionary.get(WallType.HORIZONTAL)
        vertical_wall_list = semantic_dictionary.get(WallType.VERTICAL)

        for co_ordi in horizontal_wall_list:
            start_pt, end_pt = co_ordi
            cv2.rectangle(base_image, start_pt, end_pt, (0, 114, 181), -1)

        for co_ordi in vertical_wall_list:
            start_pt, end_pt = co_ordi
            cv2.rectangle(base_image, start_pt, end_pt, (253, 172, 83), -1)

        b_folder = f'{base_folder}base_{i}.png'
        m_folder = f'{mask_folder}mask_{i}.png'
        cv2.imwrite(b_folder, base_image)
        cv2.imwrite(m_folder, mask_image)


if __name__ == '__main__':
    np.random.seed(100)
    print('Generating for Train')
    generate_floor_plan(base_folder=conf.train_folder_images,
                        mask_folder=conf.train_folder_mask,
                        num=800)
    print('Generating for Validation')
    generate_floor_plan(base_folder=conf.validation_folder_images,
                        mask_folder=conf.validation_folder_mask,
                        num=200)
    print('Generating for Test')
    generate_floor_plan(base_folder=conf.test_folder_images,
                        mask_folder=conf.test_folder_mask,
                        num=200)
