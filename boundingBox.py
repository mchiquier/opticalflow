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
    contour_map = cv2.findContours(1, 1, 2)
    num_box = contour_map.shape[0]

    # Find corner coordinates of each bounding box
    coordinates = np.zeros((num_box,4,2))
    for i in num_box:

        # Get bounding box from contour map
        (x, y, w, h) = cv2.boundingRect(contour_map[i])

        # Draw bounding box onto the original image
        cv2.rectangle(image, (x,y) ,(x+w, y+h), (0,255,0), 2)

        # Generate coordinates of current bounding rectangle and save
        this_box = np.array([
            [x, y],
            [x+w, y],
            [x, y+h],
            [x+w, y+h]
        ])
        coordinates[i,:,:] = this_box

    return image, coordinates