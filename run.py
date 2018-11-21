import cv2
import numpy as np
from scipy import misc

import estimateAllTranslation as eat
from objectTracking import objectTracking
import getFeatures as gf
from boundingBox import getBoundingBox
from applyGeometricTransformation import applyGeometricTransformation


''' Input videos and contours '''

easy = './Easy.mp4'
medium = './Medium.mp4'
hard = './hard.mp4'

easy_cnt = 'contour-easy.png'
medium_cnt = 'contour-med.png'
hard_cnt = 'contour-hard.png'

objectTracking(medium)
print("Done")
