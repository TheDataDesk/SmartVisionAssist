import patchify
from configuration import Config as conf


def patch_images(image_list=None):
    """
    Patch all the images to form images of lower resolution and store in the
    specified directory
    :param image_list: list of images to tile
    :return: image list
    """
    data_list = []
    for index in range(len(image_list)):
        image = image_list[index]
        patches_image = patchify.patchify(image,
                                          (conf.tile_height, conf.tile_width),
                                          step=conf.tile_width)
        for i in range(patches_image.shape[0]):
            for j in range(patches_image.shape[1]):
                single_patch_image = patches_image[i, j, :, :]
                single_patch_image = (single_patch_image.astype('float32')) / 255  # Value of None Background

                data_list.append(single_patch_image)

    return data_list

