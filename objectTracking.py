
import cv2
import numpy as np
from scipy import misc 
import estimateAllTranslation as eat

import getFeatures as gf
from boundingBox import getBoundingBox
from applyGeometricTransformation import applyGeometricTransformation


def objectTracking(rawVideo):

	return trackedVideo