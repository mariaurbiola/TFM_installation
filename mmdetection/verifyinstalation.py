from mmdet.apis import init_detector, inference_detector
import os
#pathname = '/home/maria/Escritorio/TFM/TFM_MariaUrbiola/mmdetection/'

#config_file = pathname+'yolov3_mobilenetv2_320_300e_coco.py'
#checkpoint_file = pathname+'yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
#model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
#inference_detector(model, pathname+'demo/demo.jpg')

print('path', os.path.dirname(__file__))
config_file = os.path.dirname(__file__) + '/yolov3_mobilenetv2_320_300e_coco.py'
checkpoint_file = os.path.dirname(__file__) + '/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
print(next(model.parameters()).device)
inference_detector(model, os.path.dirname(__file__) + '/demo/demo.jpg')
print(inference_detector(model, os.path.dirname(__file__) + '/demo/demo.jpg'))