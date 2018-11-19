import estimateFeatureTranslation as eft
import numpy as np
from scipy import signal
import cv2

def estimateAllTranslation(startXs,startYs,img1,img2):
	N,F = startXs.shape

	gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	H,W = gray_img1.shape
	Ix = signal.convolve2d(gray_img1, np.array([[-1, 0, 1]]))
	Iy = signal.convolve2d(gray_img1, np.array([[-1, 0, 1]]).transpose())
	# Iy, Ix = np.gradient(img1.astype(float))

	newXs =  np.zeros((N,F))-1
	newYs =  np.zeros((N,F))-1

	for box in range(F):
		for feat in range(N):
			startX = int(startXs[feat,box])
			startY = int(startYs[feat,box])
			if startY < 5 or startX < 5 or startY >= H-5 or startX >= W-5: continue
			newXs[feat,box],newYs[feat,box] = eft.estimateFeatureTranslation(startX, startY, Ix, Iy, gray_img1, gray_img2)
	# print(newXs.transpose())
	return newXs.astype(int), newYs.astype(int)
	