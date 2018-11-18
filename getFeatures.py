'''getFeatures.py file  '''
import numpy as np
import cv2
# from skimage.feature import corner_shi_tomasi

def getFeatures(img, bbox) : 
	N = 25
	F = bbox.shape[0]
	matX = np.zeros((N,F))
	matY = np.zeros((N,F))
	for f in range(F):
		gray = img[bbox[f,0,1]:bbox[f,2,1],bbox[f,0,0]:bbox[f,1,0]]
		response = cv2.goodFeaturesToTrack(gray,N,0.01,10).reshape((N,2))
		# if len(response.shape[0]) < N:
		# 	response = response + np.zeros((N-len(response.shape[0]),2))-1
		respNby2 = (response+bbox[f,0]).astype(int)
		matX[:,f] = respNby2[:,0]
		matY[:,f] = respNby2[:,1]
	return matX,matY 
