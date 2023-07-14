"""
Helper class to read images from a directory
"""

import os
import cv2

accepted_ext = ['png', 'jpeg']


def read_images(folder_loc):
    images_name_list = []
    images = []
    for image in sorted(os.listdir(folder_loc)):
        images_name_list.append(os.path.join(folder_loc, image))
    for img in images_name_list:
        if img.split('.')[1] in accepted_ext:
            image = cv2.imread(img, 0)
            images.append(image)

    return images
