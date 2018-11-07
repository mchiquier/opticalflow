import cv2
import numpy as np
import getFeatures

easy = './Easy.mp4'
medium = './Medium.mp4'
hard = './hard.mp4'

video = cv2.VideoCapture(easy)

success, prevframe = video.read()
prev_grayframe = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)
success, frame = video.read()

# cv2.imshow('result',frame)
# if cv2.waitKey(0) & 0xff == 27:
# 	cv2.destroyAllWindows()

while success:
	grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	
	
	# Do whatever
	
	# get next frame
	prev_grayframe = grayframe.copy()
	success, frame = video.read()

video.release()


