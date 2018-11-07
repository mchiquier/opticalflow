import cv2
import numpy as np
from scipy import misc 

import getFeatures
from boundingBox import getBoundingBox

easy = './Easy.mp4'
medium = './Medium.mp4'
hard = './hard.mp4'

video = cv2.VideoCapture(easy)

success, prevframe = video.read()
prev_grayframe = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)
success, frame = video.read()
misc.imsave('first_frame.png',prev_grayframe)
# cv2.imshow('result',frame)
# if cv2.waitKey(0) & 0xff == 27:
# 	cv2.destroyAllWindows()
count = 0 
while success:
	grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	prev_grayframe = grayframe.copy()
	success, frame = video.read()

video.release()

# Bounding box + feature detector
image = cv2.imread('first_frame.png')
contour_image = cv2.imread('contour1.png', cv2.IMREAD_GRAYSCALE)
image, bbox_corners = getBoundingBox(image, contour_image)

print(bbox_corners)
cv2.imshow("Image:", image)
if cv2.waitKey(0) & 0xff == 27:
	cv2.destroyAllWindows()

