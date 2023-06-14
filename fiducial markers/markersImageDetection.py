import cv2 as cv
from cv2 import aruco
import numpy as np
import os
print("version openCV", cv.__version__)
print("path openCV", cv.__path__)

#markerCorners = np.array([[],[]])
#x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
#markerIds = np.array
#rejectedCandidates = np.array([[],[]])

#inputImage = (cv.imread("markersToDetect.png"))
inputImage = (cv.imread(os.path.dirname(__file__) + "/imageToDetect.jpg"))
cv.imshow("Markers", inputImage)
#outputImage = inputImage


dictionary = aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250); #predefined dictionary
arucoParams = aruco.DetectorParameters_create()

corners, ids, rejectedImgPoints = aruco.detectMarkers(inputImage, dictionary, parameters=arucoParams)

#print("corners",corners)
#print("ids",ids)
#print("rejected",rejectedImgPoints)

outputImage = aruco.drawDetectedMarkers(inputImage, corners, ids, (0,255,0))

cv.imshow("Detected markers", outputImage)
cv.waitKey(0)