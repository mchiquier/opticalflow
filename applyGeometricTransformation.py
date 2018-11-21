''' applyGeometricTransformation '''

import numpy as np
from scipy.optimize import least_squares
from skimage import transform

'''
	- (INPUT) startXs: N × F
	- (INPUT) startYs: N × F
	- (INPUT) newXs: N × F
	- (INPUT) newYs: N × F
	- (INPUT) bbox: F × 4 × 2
	- (OUTPUT) Xs: N1 × F 		(N1 = number of features after filtering outliers)
	- (OUTPUT) Ys: N1 × F
	- (OUTPUT) newbbox: F × 4 × 2
'''

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
	N, F = startXs.shape[0:2]

	# Compute distance change between each pt
	diff_x = newXs - startXs
	diff_y = newYs - startYs

	distance = np.zeros((N,F))
	distance = np.sqrt(np.square(diff_x) + np.square(diff_y))
	distance[np.where(startXs == -1)] = -1

	# Filter out outliers
	threshold = 10
	where_outliers = np.where(distance >= threshold)

	startXs[where_outliers] = -1
	startYs[where_outliers] = -1
	newXs[where_outliers] = -1
	newYs[where_outliers] = -1

	Xs, Ys = newXs, newYs
	Xs = Xs[np.any(Xs != -1, axis=1)]
	Ys = Ys[np.any(Ys != -1, axis=1)]

	''' Transform bounding boxes '''
	new_bbox = np.zeros((F,4,2))
	for f in range(F):
		box = bbox[f,:,:]

		# Get transform
		this_start = np.stack(\
			(np.ravel(startXs[np.where(startXs[:,f] != -1),f]),
			np.ravel(startYs[np.where(startYs[:,f] != -1),f])), axis=1
		)
		this_new = np.stack(\
			(np.ravel(newXs[np.where(newXs[:,f] != -1),f]),
			np.ravel(newYs[np.where(newYs[:,f] != -1),f])), axis=1
		)

		# Break if all features are gone
		if (this_start.shape[0] == 0 or this_new.shape[0] == 0):
			return None, None, None

		tform = transform.estimate_transform('affine', this_start, this_new)

		# Transform the bbox
		new_bbox[f,:,:] = tform(box)

	#new_bbox = np.round(new_bbox).astype(int)
	return Xs, Ys, new_bbox









