# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser
import numpy as np
from cv2 import aruco
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
from pixelToWorld import pixelToWorld

import cv2
import mmcv

from mmpose.apis import (collect_multi_frames, get_track_id,
                         inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_tracking_result)
from mmpose.core import Smoother
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False



def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--det_config', 
        default= 'mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
        type=str, 
        help='Config file for detection')
    parser.add_argument(
        '--det_checkpoint', 
        default= os.path.dirname(__file__) + '/files_detection/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        type=str, 
        help='Checkpoint file for detection')
    parser.add_argument(
        '--pose_config', 
        default='mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py',
        type=str, 
        help='Config file for pose')
    parser.add_argument(
        '--pose_checkpoint', 
        default=os.path.dirname(__file__) + '/files_detection/res50_coco_256x192-ec54d7f3_20200709.pth',
        type=str, 
        help='Checkpoint file for pose')
    parser.add_argument(
        '--video-path', 
        default=os.path.dirname(__file__) + '/mmpose/demo/resources/demo2.mp4',
        type=str, 
        help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default=os.path.dirname(__file__) + '/videos',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--euro',
        action='store_true',
        help='(Deprecated, please use --smooth and --smooth-filter-cfg) '
        'Using One_Euro_Filter for smoothing.')
    parser.add_argument(
        '--smooth',
        action='store_true',
        help='Apply a temporal filter to smooth the pose estimation results. '
        'See also --smooth-filter-cfg.')
    parser.add_argument(
        '--smooth-filter-cfg',
        type=str,
        default='configs/_base_/filters/one_euro.py',
        help='Config file of the filter to smooth the pose estimation '
        'results. See also --smooth.')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    parser.add_argument(
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the pose'
        'estimation stage. Default: False.')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the pose'
        'estimation stage. Default: False.')
    return parser.parse_args()

def runVideoFileDetection(file_path):
    print('filepath', file_path)
    iterationNumber = 1
    distance = 100  #como incial, por si acaso en la primera frame no detecta marker
    #check if folder to store the data exists. if not, create.
    path = os.path.dirname(__file__) + '/videos/'+Path(file_path).stem+ 'Data/'
    print("folder path",path)
    if  not os.path.exists(path):
        os.mkdir(path)
        print("folder creado",path)
    
    assert has_mmdet, 'Please install mmdet'
    args = parse_args()
    args.video_path = file_path
    print("aqui")
    print(args.det_config)
    print(args.out_video_root)
    #print('assert args.show' )
    #assert args.show
    #print('args.out_video_root != ''' )
    #assert (args.out_video_root != '')

    #args = parser.parse_args()
    
    #args.det_config = 'mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
    #args.det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    #args.pose_config = 'mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py'
    #args.pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth'

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    print('Initializing model...')
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # read video
    video = mmcv.VideoReader(args.video_path)
    assert video.opened, f'Failed to load video file {args.video_path}'

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = video.fps
        size = (video.width, video.height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'result_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)
        #print("videoWriter", videoWriter['filename]'])
        videoWriterEdited = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'resultWithPoints_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)
        print("videoWriterEdited",videoWriterEdited)

    # frame index offsets for inference, used in multi-frame inference setting
    if args.use_multi_frames:
        assert 'frame_indices_test' in pose_model.cfg.data.test.data_cfg
        indices = pose_model.cfg.data.test.data_cfg['frame_indices_test']

    # build pose smoother for temporal refinement
    if args.euro:
        warnings.warn(
            'Argument --euro will be deprecated in the future. '
            'Please use --smooth to enable temporal smoothing, and '
            '--smooth-filter-cfg to set the filter config.',
            DeprecationWarning)
        smoother = Smoother(
            filter_cfg='configs/_base_/filters/one_euro.py', keypoint_dim=2)
    elif args.smooth:
        smoother = Smoother(filter_cfg=args.smooth_filter_cfg, keypoint_dim=2)
    else:
        smoother = None

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    next_id = 0
    pose_results = []
    print('Running inference...')
 

    
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        pose_results_last = pose_results
        

        # get the detection results of current frame
        # the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, cur_frame)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        if args.use_multi_frames:
            frames = collect_multi_frames(video, frame_id, indices,
                                          args.online)

        # test a single image, with a list of bboxes.
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            frames if args.use_multi_frames else cur_frame,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # get track id for each person instance
        pose_results, next_id = get_track_id(
            pose_results,
            pose_results_last,
            next_id,
            use_oks=args.use_oks_tracking,
            tracking_thr=args.tracking_thr)

        # post-process the pose results with smoother
        if smoother:
            pose_results = smoother.smooth(pose_results)

        # show the results
        vis_frame = vis_pose_tracking_result(
            pose_model,
            cur_frame,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=False)
        
        frameEdited = vis_frame.copy() #a new frame to do the changes here, but keep the original
        #these are the pixel points
        #print(pose_results)
        #code to find the distance. It is assumed that there is only 1 marker in the image
        #and that the marker will be on the person and all keypoints are at the same distance
        #with a depth camera this will not be needed and the distance will be given by the camera 
        #directñy for each point
        dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250); #predefined dictionary
        arucoParams = aruco.DetectorParameters_create()
        objPoints = (np.array([[0, 0, 1], [15, 0, 1], [15, 15, 1], [0, 15, 1]], np.int32)).astype('float32')
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frameEdited, dictionary, parameters=arucoParams)
        #print("corners", corners)
        if len(corners) == 0:
            distanceActualPoint = distance
        else:
        
            #print("corners[0][0][0]",corners[0][0][0])
            #print("corners[0][0][1]",corners[0][0][1])
            #print("corners[0][0][2]",corners[0][0][2])
            #print("corners[0][0][3]",corners[0][0][3])
            
            imgPoints = (np.array([corners[0][0][0],corners[0][0][1],corners[0][0][2],corners[0][0][3]], np.int32)).astype('float32')
            
            with open('/home/maria/Escritorio/TFM/TFM_MariaUrbiola/calibrate camera/calibration.yaml',"r") as f:
                loadeddict = yaml.load(f, Loader=SafeLoader)
                cameraMatrix = loadeddict.get('camera_matrix')
                distCoefs = loadeddict.get('dist_coeff')
                cameraMatrix = np.array(cameraMatrix)
                distCoeffs = np.array(distCoefs)
            retval, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
            #print ("Rotation ", rvec, "Translation", tvec)
            #print("retval",retval)
            distanceActualPoint = tvec[2][0]
            distance = distanceActualPoint  #update the distance for next frame
        #print("distancia ", distanceActualPoint)
        
        with open(os.path.dirname(__file__) + '/videos/'+Path(file_path).stem+'Data/Frame'+str(iterationNumber), "w") as f:
            for j in range (len(pose_results)):
                pixels = pose_results[j]['keypoints']
                for i in range (len(pixels)):
                    #convert pixel coord to world coord
                    #print("iteracion n",i+1)
                    pixeli = [pixels[i][0],pixels[i][1]]
                    #print("pose result pixel: ", pixeli)
                    worldPoint = pixelToWorld(pixeli,distanceActualPoint)
                    #print("worldPoint: ",worldPoint)
                    data = {'Person id': j,'Point Number': i+1,'Pixel coordinates': np.asarray(pixeli).tolist(), 'World coordinates': np.asarray(worldPoint).tolist()}

                    yaml.dump(data, f,sort_keys=False)
                    
                    #add the real world coordinates to the image
                    text = str((np.array(worldPoint)).astype(int))
                    frameEdited = cv2.putText(frameEdited, text, (np.array(pixeli)).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 153, 255))
                    frameEdited = cv2.circle(frameEdited,(np.array(pixeli)).astype(int), 2, (255, 153, 255), -1)
            
    
            
            '''
            for i in range (len(pixels)):
                #convert pixel coord to world coord
                #print("iteracion n",i+1)
                pixeli = [pixels[i][0],pixels[i][1]]
                #print("pose result pixel: ", pixeli)
                worldPoint = pixelToWorld(pixeli,distanceActualPoint)
                #print("worldPoint: ",worldPoint)
                data = {'Point Number': i+1,'Pixel coordinates': np.asarray(pixeli).tolist(), 'World coordinates': np.asarray(worldPoint).tolist()}

                yaml.dump(data, f,sort_keys=False)
                
                #add the real world coordinates to the image
                text = str((np.array(worldPoint)).astype(int))
                frameEdited = cv2.putText(frameEdited, text, (np.array(pixeli)).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 153, 255))
                frameEdited = cv2.circle(frameEdited,(np.array(pixeli)).astype(int), 2, (255, 153, 255), -1)
        '''
        iterationNumber = iterationNumber+1        
            
        
        

        if args.show:
            cv2.imshow('Frame', vis_frame)
            cv2.imshow('Frame Edited', frameEdited)

        if save_out_video:
            videoWriter.write(vis_frame)
            videoWriterEdited.write(frameEdited)

        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if save_out_video:
        videoWriter.release()
        videoWriterEdited.release()
    if args.show:
        cv2.destroyAllWindows()


#if __name__ == '__main__':
#    runVideoFileDetection(file_path)
    
    ##python demo/top_down_pose_tracking_demo_with_mmdet.py     demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py     https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth     configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py     https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth     --video-path demo/resources/demo.mp4     --out-video-root vis_results

