''' applyGeometricTransformation '''

import numpy as np

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
	distance[np.where(startXs != -1)] = -1

	# Filter out outliers
	threshold = 4
	where_outliers = np.where(distance >= threshold)

	'''
		From writeup:
		In the above function, you should eliminate feature points if the distance from a point to the projection of its corresponding
		point is greater than 4. You can play around with this value.
	'''

	# todo: filter out outliers and transform bboxs

	newbbox = bbox.copy()
	return Xs, Ys, newbbox

