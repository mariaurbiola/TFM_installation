import os
import cv2 as cv
import json
from cv2 import aruco
import numpy as np
from pixelToWorld import pixelToWorld
import yaml
from yaml.loader import SafeLoader
from pathlib import Path



from mmpose.apis import (init_pose_model, inference_bottom_up_pose_model, vis_pose_result)

def runImageFileDetection(file_path):

    #file_path = os.path.dirname(__file__) + '/mmpose_photos/test1.jpg'
        
    #path_to_image_folder = os.path.dirname(__file__) + '/mmpose_photos/'

    config_file = os.path.dirname(__file__) + '/files_detection/associative_embedding_hrnet_w32_coco_512x512.py'
    checkpoint_file = os.path.dirname(__file__) + '/files_detection/hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
    pose_model = init_pose_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

    #image_name = file_path + 'persons.jpg'
    # test a single image
    pose_results, other = inference_bottom_up_pose_model(pose_model, file_path)
    
    #code to find the distance. It is assumed that there is only 1 marker in the image
    #and that the marker will be on the person and all keypoints are at the same distance
    #with a depth camera this will not be needed and the distance will be given by the camera 
    #directñy for each point
    inputImage = (cv.imread(file_path))
    dictionary = aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250); #predefined dictionary
    arucoParams = aruco.DetectorParameters_create()
    objPoints = (np.array([[0, 0, 1], [15, 0, 1], [15, 15, 1], [0, 15, 1]], np.int32)).astype('float32')
    corners, ids, rejectedImgPoints = aruco.detectMarkers(inputImage, dictionary, parameters=arucoParams)
    
    if len(corners) == 0:
            distance = 100
    else:
    
        imgPoints = (np.array([corners[0][0][0],corners[0][0][1],corners[0][0][2],corners[0][0][3]], np.int32)).astype('float32')
    
        with open('/home/maria/Escritorio/TFM/TFM_MariaUrbiola/calibrate camera/calibration.yaml',"r") as f:
            loadeddict = yaml.load(f, Loader=SafeLoader)
            cameraMatrix = loadeddict.get('camera_matrix')
            distCoefs = loadeddict.get('dist_coeff')
            cameraMatrix = np.array(cameraMatrix)
            distCoeffs = np.array(distCoefs)
        retval, rvec, tvec = cv.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
        #print ("Rotation ", rvec, "Translation", tvec)
        #print("retval",retval)
        distance = tvec[2][0]
    
    imgOutName = Path(file_path).stem+"Result.jpg"
    # save and show the results without points
    imgOut = vis_pose_result(pose_model, file_path, pose_results, out_file= os.path.dirname(__file__)+'/photos/'+imgOutName)    
    cv.imshow('imagen', imgOut)
    print(pose_results)
    print(len(pose_results))

    
    with open(os.path.dirname(__file__) + '/photos/'+Path(file_path).stem, "w") as f:
        for j in range (len(pose_results)):
            pixels = pose_results[j]['keypoints']
            for i in range (len(pixels)):
                #convert pixel coord to world coord
                #print("iteracion n",i+1)
                pixeli = [pixels[i][0],pixels[i][1]]
                #print("pose result pixel: ", pixeli)
                worldPoint = pixelToWorld(pixeli,distance)
                #print("worldPoint: ",worldPoint)
                data = {'Person id': j,'Point Number': i+1,'Pixel coordinates': np.asarray(pixeli).tolist(), 'World coordinates': np.asarray(worldPoint).tolist()}

                yaml.dump(data, f,sort_keys=False)
                
                #add the real world coordinates to the image
                text = str((np.array(worldPoint)).astype(int))
                imgOut = cv.putText(imgOut, text, (np.array(pixeli)).astype(int), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 153, 255))
                imgOut = cv.circle(imgOut,(np.array(pixeli)).astype(int), 2, (255, 153, 255), -1)
                
    
    # save and show the results with points
    imgOutName = os.path.dirname(__file__)+'/photos/'+Path(file_path).stem+"ResultWithPoints.jpg"
    cv.imwrite(imgOutName, imgOut)
    cv.imshow('imagen con puntos', imgOut)
    cv.waitKey(0)