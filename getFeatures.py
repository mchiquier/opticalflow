'''getFeatures.py file  '''
import cv2
from skimage.feature import corner_shi_tomasi

def getFeatures(img, bbox) : 
    image_response = feature.corner_shi_tomasi(img)
    corners_in_bbox = image_response[bbox]
    return corners_in_bbox
