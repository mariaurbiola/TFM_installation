import cv2 as cv
import os
#print(cv.__version__)
#print(os.path.dirname(__file__))

#Predefined dictionary
predefinedDictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250)  #there are many more dictionaries but I'm using this
sidePixels = 200
borderBits = 1

#To create several markers
numberOfMarkers = 10
for i in range(numberOfMarkers):
    markerImage = cv.aruco.drawMarker(predefinedDictionary, i, sidePixels, borderBits)
    cv.imwrite(os.path.dirname(__file__) + "/markers/markerPredefined"+str(i)+".png", markerImage)

'''
#To create a specific marker given an id
id = 3  #id of the marker desired
markerImage = cv.aruco.drawMarker(predefinedDictionary, id, sidePixels, borderBits)
cv.imwrite(os.path.dirname(__file__) + "/markers/markerPredefined"+str(id)+".png", markerImage)
'''


''' Automatic dictionary is, as far as I'm concerned, only available in the new version of 
    OpenCV, 4.7.0.
    I'm currently using version 4.6.0 because the newest version is not stable with aruco.
    Because of that, this section is ignored

#Automatic dictionary
numberOfMarkers = 6
numberOfBits = 5
for i in range(numberOfMarkers):
    automaticDictionary = cv.aruco.extendDictionary(numberOfMarkers, numberOfBits)
    markerImage = cv.aruco.drawMarker(automaticDictionary, i, sidePixels, borderBits)

    cv.imwrite("markers/markerAutomatic"+str(i)+".png", markerImage)
'''

