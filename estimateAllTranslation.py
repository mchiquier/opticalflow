import estimateFeatureTranslation as eft
import numpy as np

def estimateAllTranslation(startXs,startYs,img1,img2):
	N,F = startXs.shape

	Ix = scipy.signal.convolve2D(img1, np.array([1, 0, -1]))
	Iy = scipy.signal.convolve2D(img1, np.array([1, 0, -1]).transpose())
	# dy, dx = np.gradient(img.astype(float))

	newXs =  np.zeros((N,F))-1
	newYs =  np.zeros((N,F))-1

	for box in range(F):
		for feat in range(N):
			startX = startXs[feat,box]
			startY = startYs[feat,box]
			if startY < 0 or startX < 0: continue
			newXs[feat,box],newYs[feat,box] = eft.estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2)			
	return newXs, newYs
	