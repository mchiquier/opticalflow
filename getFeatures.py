'''getFeatures.py file  '''
import numpy as np
import cv2
# from skimage.feature import corner_shi_tomasi

def getFeatures(img, bbox):
	N = 20
	F = bbox.shape[0]
	bbox_round = np.round(bbox).astype(int)

	# initialized for -1s
	matX = np.zeros((N,F))-1
	matY = np.zeros((N,F))-1
	for f in range(F):
		# Get responsive locations within bounding box
		gray = img[bbox_round[f,0,1]:bbox_round[f,2,1],bbox_round[f,0,0]:bbox_round[f,1,0]]
		response = cv2.goodFeaturesToTrack(gray,N,0.01,10)

		if (response is None):
			return None, None
		response = response.reshape((response.shape[0],2))

		# [x y] response matrix; add top left corner to response position
		n = response.shape[0] # between 0 and N
		respNby2 = (response+bbox[f,0]).astype(int)
		matX[0:n,f] = respNby2[:,0]
		matY[0:n,f] = respNby2[:,1]

	return matX,matY
