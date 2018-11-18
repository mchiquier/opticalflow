'''
    boundingBox.py
'''


import numpy as np
import cv2

''' getBoundingBox:
 - input: image - original image to draw bounding box on
 - input: contour_image - binary image denoting contour
'''
def getBoundingBox(image, contour_image):

    # Get bounding box from contour map
    (x, y, w, h) = cv2.boundingRect(contour_image)

    # Draw bounding box onto the original image
    cv2.rectangle(image, (x,y) ,(x+w, y+h), (0,255,0), 2)

    # Generate coordinates of the bounding rectangle and return
    coordinates = np.array([
        [x, y],
        [x+w, y],
        [x, y+h],
        [x+w, y+h]
    ])

    return image, coordinates