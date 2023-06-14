import os
import json
import numpy as np
from pixelToWorld import pixelToWorld
print('path', os.path.dirname(__file__))

from mmpose.apis import (init_pose_model, inference_bottom_up_pose_model, vis_pose_result)

def runImageFileDetection(file_path):

    #file_path = os.path.dirname(__file__) + '/mmpose_photos/test1.jpg'
        
    #path_to_image_folder = os.path.dirname(__file__) + '/mmpose_photos/'

    config_file = os.path.dirname(__file__) + '/mmpose/associative_embedding_hrnet_w32_coco_512x512.py'
    checkpoint_file = os.path.dirname(__file__) + '/mmpose/hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
    pose_model = init_pose_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

    #image_name = file_path + 'persons.jpg'
    # test a single image
    pose_results, other = inference_bottom_up_pose_model(pose_model, file_path)


    pixels = pose_results[0]['keypoints']


    for i in range (len(pixels)):
        print("iteracion n",i+1)
        pixeli = [pixels[i][0],pixels[i][1]]
        print("pose result pixel: ", pixeli)
        worldPoint = pixelToWorld(pixeli)
        print("worldPoint: ",worldPoint)
            

    # show the results
    vis_pose_result(pose_model, file_path, pose_results, out_file= os.path.dirname(__file__)+'/mmpose_photos/result2.jpg')    #cambiar esto porque lo guarda en home