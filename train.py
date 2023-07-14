"""
@Author Sujith Umapathy

Script to train a semantic segmentation model

Train on manually generated floor plans and their corresponding labels
"""

import os
import tensorflow as tf
from utils.read_images import read_images
from configuration import Config as conf
from utils.visualise import visualize
import numpy as np
from utils.patch_images import patch_images
import keras.backend as k
import math

k.set_image_data_format('channels_last')

# need to choose keras backend for the library to work
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm


def clean_list(list_data):
    mod_list = []
    for im, ma in list_data:
        if ma.any():
            mod_list.append((im, ma))
    return mod_list


if __name__ == '__main__':
    # Read train images
    train_images = read_images(conf.train_folder_images)
    train_mask = read_images(conf.train_folder_mask)
    # Read validate images
    val_images = read_images(conf.validation_folder_images)
    val_mask = read_images(conf.validation_folder_mask)

    print(f'Length of Train images {len(train_images)}')
    print(f'Length of Validation images {len(val_images)}')

    visualize(train_images, train_mask)

    visualize(val_images, val_mask)

    train_images = np.array(train_images)
    train_mask = np.array(train_mask)

    val_images = np.array(val_images)
    val_mask = np.array(val_mask)

    # patch_images
    t_train_images = patch_images(train_images)
    t_train_masks = patch_images(train_mask)

    t_val_images = patch_images(val_images)
    t_val_masks = patch_images(val_mask)

    t_train_images = np.array(t_train_images)
    t_train_masks = np.array(t_train_masks)

    t_val_images = np.array(t_val_images)
    t_val_masks = np.array(t_val_masks)

    # remove useless frames
    t_list = clean_list(zip(t_train_images, t_train_masks))
    v_list = clean_list(zip(t_val_images, t_val_masks))

    t_train_images, t_train_masks = zip(*t_list)
    t_val_images, t_val_masks = zip(*v_list)

    t_train_images = np.array(t_train_images)
    t_train_masks = np.array(t_train_masks)

    t_val_images = np.array(t_val_images)
    t_val_masks = np.array(t_val_masks)

    # stack layers for aligning to segmentation model - Keras requirement of 4 dimensions
    t_train_images = np.stack((t_train_images,) * 3, axis=-1)
    t_val_images = np.stack((t_val_images,) * 3, axis=-1)

    t_train_masks = np.expand_dims(t_train_masks, axis=-1)
    t_val_masks = np.expand_dims(t_val_masks, axis=-1)

    n_classes = np.unique(t_train_masks)

    backbone = conf.backbone
    pre_process_ip = sm.get_preprocessing(backbone)

    t_train_images = pre_process_ip(t_train_images)
    t_val_images = pre_process_ip(t_val_images)

    # Train model

    model = sm.Unet(backbone, encoder_weights='imagenet')

    model.compile(optimizer='Adam',
                  loss=sm.losses.binary_focal_jaccard_loss,
                  metrics=sm.metrics.IOUScore(threshold=0.5))

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f'{conf.model_path}model_resnet_18.h5',
            save_weights_only=True,
            save_best_only=True,
            mode='auto',
            monitor='loss'),

        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                             patience=3, min_lr=0.00001)
    ]

    history = model.fit(
        t_train_images, t_train_masks,
        steps_per_epoch=200,
        epochs=20,
        callbacks=callbacks,
        validation_data=(t_val_images, t_val_masks),
        validation_steps=math.ceil(t_val_images.shape[0] / 10),
        batch_size=10
    )
