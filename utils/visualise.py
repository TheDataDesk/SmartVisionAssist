"""
Helper class to visualise the image and mask correctness during train phase
"""

import matplotlib.pyplot as plt
import numpy as np


def visualize(image_list, mask_list):
    num = np.random.randint(0, len(image_list), 1)[0]
    image = image_list[num]
    mask = mask_list[num]

    plt.imshow(image,cmap='gray')
    plt.imshow(mask,cmap='hot',alpha=0.4)
    plt.show()
