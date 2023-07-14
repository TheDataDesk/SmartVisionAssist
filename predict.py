"""
@Author Sujith Umapathy

Helper class to perform segmentation of a floor plan
"""

import os

import cv2
import keras.backend as k

k.set_image_data_format('channels_last')

import numpy as np

from utils.patch_images import patch_images

from patchify import unpatchify
from configuration import Config as conf

os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm

backbone = conf.backbone
pre_process_ip = sm.get_preprocessing(backbone)


def predict_obstacles(floorplan_path):
    """
    1. Tile the images
    2. Predict each image tile
    3. Combine the predictions and return the completed image

    :param floorplan_path: path of the image to perform segmentation
    :return: segmented image
    """
    print('Reading Image')
    image = cv2.imread(floorplan_path, 0)
    image = cv2.resize(image, (conf.floor_plan_width, conf.floor_plan_height), interpolation=cv2.INTER_NEAREST)
    t_test_images = patch_images([image])

    t_test_images = np.stack((t_test_images,) * 3, axis=-1)

    t_test_images = pre_process_ip(t_test_images)

    model = sm.Unet(backbone, encoder_weights='imagenet')

    model_path = f'{conf.model_path}model_resnet_18.h5'

    print('Loading Model')
    model.load_weights(model_path)

    model.compile(optimizer='Adam',
                  loss=sm.losses.binary_focal_jaccard_loss,
                  metrics=sm.metrics.IOUScore(threshold=0.5))

    print('Predicting')
    pred_list = model.predict(t_test_images)

    pred_list_reshape = np.reshape(pred_list, (3, 4, 256, 256))
    final_image = unpatchify(pred_list_reshape, image.shape)

    return final_image
