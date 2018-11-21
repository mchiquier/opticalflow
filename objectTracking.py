
import cv2
import numpy as np
from scipy import misc
import estimateAllTranslation as eat
from maskImage import maskImage

import getFeatures as gf
from boundingBox import getBoundingBox
from applyGeometricTransformation import applyGeometricTransformation

'''
	- (INPUT) rawVideo: filename of the input video
	- (OUTPUT) trackedVideo:
'''

def objectTracking(rawVideo):

	video = cv2.VideoCapture(rawVideo)

	success, prevframe = video.read()
	prev_grayframe = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)

	''' Get contour image '''
	num_objects = int(input("How many objects to track? : "))
	all_bbox_corners = np.zeros((num_objects,4,2))
	for f in range(num_objects):
		contour_image = maskImage(prev_grayframe).astype(np.uint8)
		image, bbox_corners = getBoundingBox(prev_grayframe, contour_image)
		all_bbox_corners[f,:,:] = bbox_corners

	xs,ys = gf.getFeatures(prev_grayframe, all_bbox_corners)

	# first_frame = prevframe.copy()
	# first_frame[xs, ys, :] = np.array([0, 0, 255])

	''' Open up output file '''
	w =int(video.get(3))
	h = int(video.get(4))
	fps = video.get(cv2.CAP_PROP_FPS)
	out = cv2.VideoWriter("out.avi",cv2.VideoWriter_fourcc(*'MJPG'), fps, (w,h))

	count = 0 	# to count out the iteration
	while 1:
		prev_grayframe = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)

		success, frame = video.read()
		if not success: break
		grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Every 6 frames, get new features
		if count % 6 == 0:
			print("iter", count)
			more_xs, more_ys = gf.getFeatures(grayframe, all_bbox_corners)
			if more_xs is not None:
				xs = np.concatenate((xs, more_xs), axis=0)
				ys = np.concatenate((ys, more_ys), axis=0)
			#print(xs.shape, ys.shape)

		newXs, newYs = eat.estimateAllTranslation(xs, ys, prevframe, frame)
		finalXs, finalYs, final_bbox = applyGeometricTransformation(xs, ys, newXs, newYs, all_bbox_corners)
		if finalXs is None:
			xs, ys = gf.getFeatures(grayframe, all_bbox_corners)
			newXs, newYs = eat.estimateAllTranslation(xs, ys, prevframe, frame)
			finalXs, finalYs, final_bbox = applyGeometricTransformation(xs, ys, newXs, newYs, all_bbox_corners)

		output = frame.copy()

		for f in range(num_objects):
			x  = int(np.round(final_bbox[f,0,0]))
			xw = int(np.round(final_bbox[f,3,0]))
			y  = int(np.round(final_bbox[f,0,1]))
			yh = int(np.round(final_bbox[f,3,1]))
			cv2.rectangle(output, (x,y) ,(xw, yh), (0,255,0), 2)
		out.write(output)

		prevframe = frame.copy()
		xs, ys, all_bbox_corners = finalXs, finalYs, final_bbox

		count += 1
	print("done loop")

	video.release()
	out.release()

	return out