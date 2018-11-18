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
w =int(video.get(3))
h = int(video.get(4))
fps = video.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc(*'MJPG'), fps, (w,h))

success, prevframe = video.read()
prev_grayframe = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)

contour_image = cv2.imread('contour1.png', cv2.IMREAD_GRAYSCALE)
image, bbox_corners = getBoundingBox(prev_grayframe, contour_image)
all_bbox_corners = bbox_corners.reshape((1,4,2))
xs,ys = gf.getFeatures(prev_grayframe, all_bbox_corners)

print(all_bbox_corners)
x,y,xw,yh = 295,190,398,264

while 1:
	prev_grayframe = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)

	success, frame = video.read()
	if not success: break
	grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	newXs, newYs = eat.estimateAllTranslation(xs, ys, prevframe, frame)
	
	u,v = int(np.mean(newXs)-np.mean(xs)),int(np.mean(newYs)-np.mean(ys))
	x += u
	xw += u
	y += v
	yh += v
	output = frame.copy()
	cv2.rectangle(output, (x,y) ,(xw, yh), (0,255,0), 2)
	out.write(output)

	prevframe = frame.copy()
	xs,ys = newXs, newYs

video.release()
out.release()
# Bounding box + feature detector


cv2.imshow("Image:", image)
if cv2.waitKey(0) & 0xff == 27:
	cv2.destroyAllWindows()

