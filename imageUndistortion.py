import cv2 as cv
from cv2 import aruco
import os
import yaml
from yaml.loader import SafeLoader
import numpy as np
from pixelToWorld import pixelToWorld

#ger aurco dictionary
dictionary = aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250); #predefined dictionary
arucoParams = aruco.DetectorParameters_create()

#get camer matrix and distortion coefficients from file (previously calibrated)
with open(os.path.dirname(__file__) + '/calibrate camera/calibration.yaml',"r") as f:
    loadeddict = yaml.load(f, Loader=SafeLoader)
cameraMatrix = loadeddict.get('camera_matrix')
distCoefs = loadeddict.get('dist_coeff')
cameraMatrix = np.array(cameraMatrix)
distCoeffs = np.array(distCoefs)
print(cameraMatrix)

#input image, find markers
imgIn = cv.imread(os.path.dirname(__file__) + '/calibrate camera/aruco_data_new/imgEditada.jpg')
imgInCopy = imgIn
#corners, ids, rejectedImgPoints = aruco.detectMarkers(imgInCopy, dictionary, parameters=arucoParams)
#imgInMarkers = aruco.drawDetectedMarkers(imgInCopy, corners, ids, (0,255,0))
#if (ids is not None):
#    cv.imshow("Input image", imgInMarkers)
#print(ids)
cv.imshow("Input image", imgIn)

# undistort image, find markers
im_gray = cv.cvtColor(imgInCopy,cv.COLOR_RGB2GRAY)
h,  w = im_gray.shape[:2]
newCameraMatrix, roi=cv.getOptimalNewCameraMatrix(cameraMatrix,distCoeffs,(w,h),1,(w,h))
imgOut = cv.undistort(imgIn, cameraMatrix, distCoeffs, None, newCameraMatrix)
#imgOut = cv.undistort(imgDst, cameraMatrix, distCoeffs, None, None)
corners, ids, rejectedImgPoints = aruco.detectMarkers(imgOut, dictionary, parameters=arucoParams)
imgOutMarkers = aruco.drawDetectedMarkers(imgOut, corners, ids, (0,255,0))
#if (ids2 is not None):    
#    cv.imshow("Output image", imgOutMarkers)
print("new camera matrix", newCameraMatrix)
cv.imshow("Output image", imgOutMarkers)
cv.waitKey(0)

markerLength = 150  # Here, measurement unit is milimetre. 
markerSeparation = 700   # Here, measurement unit is milimetre. 
#board = aruco.GridBoard_create(2, 2, markerLength, markerSeparation, dictionary)

#objPoints, imgPoints = aruco.getBoardObjectAndImagePoints(board, corners, ids)	
#print("objeto al principio",objPoints)
#print("imagen al principio",imgPoints)

#retval, rvec, tvec = cv.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs) 
#print(retval)
#print(rvec)
#print(tvec)

#ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, None, None) # For a board
#print ("ret ",ret,"Rotation ", rvec, "Translation", tvec)

rvecs, tvecs, objPoints = aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs)
print("rvecs",rvecs)
print("tvecs",tvecs[0][0])
print("objPOints",objPoints)
print("corners",corners)
#y ahora se supone que esa es la traslacion entre camera world y
##getBoardObjectAndImagePoints()
for i in range (len(tvecs)):
    #print(i)
    imgOutMarkers = cv.drawFrameAxes(imgOutMarkers, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 50)


rvec = rvecs
tvec = tvecs[0][0]

Rmatrix, jacobiano = cv.Rodrigues(rvec)
#print(Rmatrix)

print("R|t - Extrinsic Matrix")
RTmatrix = np.column_stack((Rmatrix,tvec))
print(RTmatrix)
arrayAux = np.array([0, 0, 0, 1])
#print(arrayAux)
#print(np.array(RTmatrix))
matrixRTAux = np.row_stack((RTmatrix,arrayAux))


print("newCamMtx*R|t - Projection Matrix")
projectionMatrix = cameraMatrix.dot(RTmatrix)
projectionMatrix = cameraMatrix @ RTmatrix
print(projectionMatrix)
projectionMatrixExt = np.row_stack((projectionMatrix,arrayAux))

pixelPointMatrixExt = (np.array([corners[0][0][0][0],corners[0][0][0][1],1,1])).reshape(-1,1)

worldPointMatrixDirect = np.linalg.inv(projectionMatrixExt) @ pixelPointMatrixExt
print("point direct",worldPointMatrixDirect)



pixelPointMatrix = (np.array([corners[0][0][0][0],corners[0][0][0][1],1])).reshape(-1,1)

pixelPointTest = (np.array([400,400,1])).reshape(-1,1)
cv.circle(imgOutMarkers,(208,366), 5, (0,0,255), -1)    #cx,cy en rojo
cv.circle(imgOutMarkers,(400,400), 5, (255,0,255), -1)  #pixel a encontrar, en rosa
print("pixel point", pixelPointTest)
matrixRTAuxInv = np.linalg.inv(matrixRTAux)
#print("inversa", matrixRTAuxInv)

cameraMatrixInv = np.linalg.inv(cameraMatrix)
imagePlanePointMatrix = cameraMatrixInv @ pixelPointTest

print("image plane point ", imagePlanePointMatrix)

cameraPointMatrixExtended = np.row_stack((imagePlanePointMatrix,1))
print(cameraPointMatrixExtended)
worldPointMatrix = matrixRTAuxInv @ cameraPointMatrixExtended
print("point",worldPointMatrix)

distancia = 3000
worldPointMatrix2 = imagePlanePointMatrix * distancia
print("World point new",worldPointMatrix2)

#worldPointMatrix y worldPointMatrixDirect son iguales, como debe ser. 
#pero no les encuentro sentido a los resultados :(
    
    
#pruebo otra cosa
#t = cambia algo para el git
result = pixelToWorld([400,400])
print("result form function: ",result)
cv.imshow("Output image Axis", imgOutMarkers)
cv.waitKey(0)