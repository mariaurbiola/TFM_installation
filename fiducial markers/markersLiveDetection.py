import cv2 as cv
#from cv2 import aruco
import numpy as np

print(cv.__version__)
print(cv.__path__)

dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250); #predefined dictionary
arucoParams = cv.aruco.DetectorParameters_create()

inputVideo = cv.VideoCapture(0)
ret, inputImage = inputVideo.read()
                            
while True :
    
    if cv.waitKey(20) & 0xFF != ord('q'):

        ret, inputImage = inputVideo.read()
        cv.imshow("img", inputImage)
        corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(inputImage, dictionary,parameters=arucoParams)
        
        # if at least one marker detected
        if (ids is not None):
            outputImage = cv.aruco.drawDetectedMarkers(inputImage, corners, ids, (0,255,0))
            cv.imshow("Detected markers", outputImage)  #this optput image is refreshing in real time as long as there areids
        else:
            cv.imshow("Detected markers", inputImage)
    else:
        break