import cv2
import numpy as np
from scipy import misc 
import estimateAllTranslation as eat

import getFeatures as gf
from boundingBox import getBoundingBox

easy = './Easy.mp4'
medium = './Medium.mp4'
hard = './hard.mp4'

video = cv2.VideoCapture(easy)

success, prevframe = video.read()
prev_grayframe = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)
# misc.imsave('first_frame.png',prev_grayframe)
# cv2.imshow('result',frame)
# if cv2.waitKey(0) & 0xff == 27:
# 	cv2.destroyAllWindows()
count = 0 
# image = cv2.imread('first_frame.png')
contour_image = cv2.imread('contour1.png', cv2.IMREAD_GRAYSCALE)
image, bbox_corners = getBoundingBox(prev_grayframe, contour_image)
all_bbox_corners = bbox_corners.reshape((1,4,2))
xs,ys = gf.getFeatures(prev_grayframe, all_bbox_corners)
while success:
	prev_grayframe = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)

	success, frame = video.read()
	grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	xs, ys = eat.estimateAllTranslation(xs, ys, prevframe, frame)
	print(np.mean(xs),np.mean(ys))
	prevframe = frame.copy()

video.release()

# Bounding box + feature detector


cv2.imshow("Image:", image)
if cv2.waitKey(0) & 0xff == 27:
	cv2.destroyAllWindows()

