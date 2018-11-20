import cv2
import numpy as np
from scipy import misc 
import estimateAllTranslation as eat

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

video = cv2.VideoCapture(easy)
contour_image = cv2.imread(easy_cnt, cv2.IMREAD_GRAYSCALE)


w =int(video.get(3))
h = int(video.get(4))
fps = video.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc(*'MJPG'), fps, (w,h))

success, prevframe = video.read()
prev_grayframe = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)

# cv2.imwrite('first-easy.png',prev_grayframe)

image, bbox_corners = getBoundingBox(prev_grayframe, contour_image)
all_bbox_corners = bbox_corners.reshape((1,4,2))
xs,ys = gf.getFeatures(prev_grayframe, all_bbox_corners)

print(all_bbox_corners)
x,y,xw,yh = bbox_corners[0,0],bbox_corners[0,1],bbox_corners[3,0],bbox_corners[3,1]

count = 1 	# to count out the iteration
while 1:
	print("iter", count)
	prev_grayframe = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)

	success, frame = video.read()
	if not success: break
	grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Every 10 frames, get new features
	if count % 20 == 0:
		xs, ys = gf.getFeatures(grayframe, all_bbox_corners)

	newXs, newYs = eat.estimateAllTranslation(xs, ys, prevframe, frame)

	finalXs, finalYs, final_bbox = applyGeometricTransformation(xs, ys, newXs, newYs, all_bbox_corners)

	# ind = np.where(newXs != -1)
	# if newXs[ind].shape[0] > 0:
	# 	u,v = int(np.mean(newXs[ind]-xs[ind])),int(np.mean(newYs[ind]-ys[ind]))
	# 	x += u
	# 	xw += u
	# 	y += v
	# 	yh += v

	x  = final_bbox[0,0,0]
	xw = final_bbox[0,3,0]
	y  = final_bbox[0,0,1]
	yw = final_bbox[0,3,1]
	output = frame.copy()
	cv2.rectangle(output, (x,y) ,(xw, yh), (0,255,0), 2)
	out.write(output)

	prevframe = frame.copy()
	xs, ys, all_bbox_corners = finalXs, finalYs, final_bbox

	count += 1
print("done loop")

video.release()
out.release()
# Bounding box + feature detector


cv2.imshow("Image:", image)
if cv2.waitKey(0) & 0xff == 27:
	cv2.destroyAllWindows()