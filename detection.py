from webcamDetection import runWebcamDetection
from imageFileDetection import runImageFileDetection
from videoFileDetection import runVideoFileDetection
import os
import cv2
from argparse import ArgumentParser
print(cv2.__version__)
print(cv2.__path__)



def detection(detection_mode, media_file):
    '''
    parser = ArgumentParser()
    parser.add_argument(
        '--detection_mode', 
        default= 'webcam',
        type=str, 
        help='Mode to detect: webcam, imageFile or videoFile')
    parser.add_argument(
        '--media_file', 
        default= 'persons.jpg',
        type=str, 
        help='Media file to detect')
    
    args = parser.parse_args()
    
    if args.detection_mode == 'webcam':
        print('webcam')
        runWebcamDetection()
    elif args.detection_mode == 'imageFile':
        print('imageFile')
        path_to_media_folder = os.path.dirname(__file__) + '/mmpose_photos/'
        media_name = path_to_media_folder + args.media_file #change this with the name of the file
        runImageFileDetection(media_name)
    elif args.detection_mode == 'videoFile':
        print('videoFile')
        path_to_media_folder = os.path.dirname(__file__) + '/mmpose_videos/'
        media_name = path_to_media_folder + args.media_file #change this with the name of the file
        runVideoFileDetection(media_name)
    else:
        print('nada')
    '''
    if detection_mode == 'webcam':
        print('webcam')
        runWebcamDetection()
    elif detection_mode == 'imageFile':
        print('imageFile')
        path_to_media_folder = os.path.dirname(__file__) + '/mmpose_photos/'
        media_name = path_to_media_folder + media_file #change this with the name of the file
        runImageFileDetection(media_name)
    elif detection_mode == 'videoFile':
        print('videoFile')
        path_to_media_folder = os.path.dirname(__file__) + '/mmpose_videos/'
        media_name = path_to_media_folder + media_file #change this with the name of the file
        runVideoFileDetection(media_name)
    else:
        print('nada')
        
        
detection_mode = 'imageFile'
media_file = 'test1.jpg'
#detection_mode = 'videoFile'
#media_file = 'demo.mp4'
#detection_mode = 'webcam'
detection(detection_mode, media_file)