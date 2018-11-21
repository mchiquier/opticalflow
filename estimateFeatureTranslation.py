import numpy as np
import interp

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):

	prev_diff = float("inf")
	su,sv = 0,0

	meshy, meshx = np.meshgrid(np.arange(startY-5,startY+5), np.arange(startX-5, startX+5))
	img2_window = img2[startY-5:startY+5,startX-5:startX+5]

	while 1:
		It = img2 - img1

		pIx = interp.interp2(Ix, meshx, meshy)
		pIy = interp.interp2(Iy, meshx, meshy)
		pIt = interp.interp2(It, meshx, meshy)
		
		xx = np.sum(np.multiply(pIx,pIx))
		xy = np.sum(np.multiply(pIx,pIy))
		yy = np.sum(np.multiply(pIy,pIy))
		xt = np.sum(np.multiply(pIx,pIt))
		yt = np.sum(np.multiply(pIy,pIt))

		b = - np.array([xt,yt]).transpose()
		A = np.array([[xx+10e-6,xy],[xy,yy+10e-6]])
		uv = np.matmul(np.linalg.inv(A),b)

		meshy,meshx = meshy+(uv[1]),meshx+(uv[0])
		
		newImg = interp.interp2(img1, meshx, meshy)
		diff = np.linalg.norm(img2_window - newImg)
		if (prev_diff-diff) < 10e-6:
			# return np.round(startX + su), np.round(startY + sv)
			return startX + su, startY + sv
			# return np.round(startX + uv[0]), np.round(startY + uv[1])
		else:
			su += uv[0]
			sv += uv[1]
			prev_diff = diff
