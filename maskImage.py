'''
  File name: maskImage.py
  Author: John Wallison
'''

# libs
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.mlab import dist_point_to_segment
from matplotlib.path import Path

# files
from maskCreator import MaskCreator

def createMask(image_gray, actual_image):
    ax = plt.subplot(111)
    ax.imshow(image)

    mc = MaskCreator(ax)
    plt.show()

    mask = mc.get_mask(image.shape)
    image[~mask] = np.uint8(np.clip(image[~mask] - 100., 0, 255))
    plt.imshow(image, cmap="gray")
    plt.title('Region outside of mask is darkened')
    plt.show()
    return mask

def maskImage(image_gray):

    # use MaskCreator to get the mask
    ax = plt.subplot(111)
    ax.imshow(image_gray, cmap="gray")
    mc = MaskCreator(ax)
    plt.show()

    # return mask
    mask = mc.get_mask(image_gray.shape)
    return mask