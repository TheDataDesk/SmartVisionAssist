import os
from pathlib import Path


class Config:
    root = Path(os.path.dirname(__file__))

    train_folder_images = f'{root}/data/raw/train/base/'
    train_folder_mask = f'{root}/data/raw/train/mask/'

    validation_folder_images = f'{root}/data/raw/validation/base/'
    validation_folder_mask = f'{root}/data/raw/validation/mask/'

    test_folder_images = f'{root}/data/raw/test/base/'
    test_folder_mask = f'{root}/data/raw/test/mask/'

    model_path = f'{root}/model/'

    asset_path = f'{root}/asset/'

    output_path = f'{root}/output/'

    object_detection_path = f'{root}/object_detection/'

    floor_plan_width = 1024
    floor_plan_height = 768

    width_max_limit = 350
    width_min_limit = 20

    tile_height = 256
    tile_width = 256

    step_size = 20

    backbone = 'resnet18'


