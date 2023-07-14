"""
@Author Sujith Umapathy
Code to evaluate the performance of the trained model
"""
import os

import keras.backend as K
import numpy as np

from configuration import Config as conf
from utils.patch_images import patch_images
from utils.read_images import read_images

K.set_image_data_format('channels_last')
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm


def clean_list(list_data):
    mod_list = []
    for im, ma in list_data:
        if ma.any():
            mod_list.append((im, ma))
    return mod_list


if __name__ == '__main__':
    backbone = conf.backbone
    pre_process_ip = sm.get_preprocessing(backbone)

    print('Reading Images from test folder')
    test_images = read_images(conf.test_folder_images)
    test_mask = read_images(conf.test_folder_mask)

    test_images = np.array(test_images)
    test_mask = np.array(test_mask)

    print('Patching Test Image Set')

    # patch_images
    t_test_images = patch_images(test_images)
    t_test_masks = patch_images(test_mask)

    t_test_images = np.stack((t_test_images,) * 3, axis=-1)

    t_test_masks = np.expand_dims(t_test_masks, axis=-1)

    t_test_images = pre_process_ip(t_test_images)

    model = sm.Unet(backbone, encoder_weights='imagenet')

    model.load_weights(f'{conf.model_path}model_resnet_18.h5')

    model.compile(optimizer='Adam',
                  loss=sm.losses.binary_focal_jaccard_loss,
                  metrics=sm.metrics.IOUScore(threshold=0.5))

    prediction_set = model.predict(t_test_images[:600])
    score = model.evaluate(t_test_images[:600], prediction_set)

    print(f'Test score: {score}')
