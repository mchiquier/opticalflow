'''getFeatures.py file  '''
import cv2
# from skimage.feature import corner_shi_tomasi

def getFeatures(img, bbox) : 
    gray = cv2.cvtColor(img[bbox[0,1]:bbox[2,1],bbox[0,0]:bbox[1,0]], cv2.COLOR_BGR2GRAY)
    response = cv2.goodFeaturesToTrack(gray,25,0.01,10).reshape((25,2))
    return response+bbox[0]
