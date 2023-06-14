import cv2 as cv
from cv2 import aruco
import os
import yaml
from yaml.loader import SafeLoader
import numpy as np

def pixelToWorld(pixelPoint):

    #get camer matrix and distortion coefficients from file (previously calibrated)
    with open(os.path.dirname(__file__) + '/calibrate camera/calibration.yaml',"r") as f:
        loadeddict = yaml.load(f, Loader=SafeLoader)
    cameraMatrix = loadeddict.get('camera_matrix')
    distCoefs = loadeddict.get('dist_coeff')
    cameraMatrix = np.array(cameraMatrix)
    distCoeffs = np.array(distCoefs)
    #print("Intinsic matrix",cameraMatrix)

    cameraMatrixInv = np.linalg.inv(cameraMatrix)



    pixelPointMatrix = (np.array([pixelPoint[0],pixelPoint[1],1])).reshape(-1,1)
    #print("pixel point", pixelPointMatrix)


    imagePlanePointMatrix = cameraMatrixInv @ pixelPointMatrix

    #print("image plane point ", imagePlanePointMatrix)

    distancia = 3000    #la distancia al punto, que idealmente hubiera sido con la camara suya
    worldPointMatrix = imagePlanePointMatrix * distancia
    #print("World point new",worldPointMatrix)
    
    return(worldPointMatrix)
