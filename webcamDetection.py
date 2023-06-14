import os
import logging
from argparse import ArgumentParser

from mmcv import Config, DictAction

from mmpose.apis.webcam import WebcamExecutor
from mmpose.apis.webcam.nodes import model_nodes




def parse_args():
    parser = ArgumentParser('Webcam executor configs')
    parser.add_argument(
        '--config', type=str, default='mmpose/demo/webcam_cfg/pose_estimation.py')  #he editado el path, empezaba en demo

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='Override settings in the config. The key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options executor_cfg.camera_id=1'")
    
    parser.add_argument(
        '--cpu', action='store_true', default=True, help='Use CPU for model inference.')

    return parser.parse_args()


def set_device(cfg: Config, device: str):
    """Set model device in config.
    Args:
        cfg (Config): Webcam config
        device (str): device indicator like "cpu" or "cuda:0"
    """
    device = device.lower() #devuelve la string en minusculas
    assert device == 'cpu' or device.startswith('cuda:')    #comprueba la condicion, si no se da para el programa

    for node_cfg in cfg.executor_cfg.nodes:
        if node_cfg.type in model_nodes.__all__:
            #print('antes')
            #print(node_cfg)
            node_cfg.update(device=device)
            
        if node_cfg.type == 'DetectorNode':
            model_configFile = os.path.dirname(__file__) + '/mmpose/demo/mmdetection_cfg/ssdlite_mobilenetv2_scratch_600e_coco.py'
            node_cfg.model_config = model_configFile
    return cfg


def runWebcamDetection():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    if args.cpu:
        cfg = set_device(cfg, 'cpu')

    webcam_exe = WebcamExecutor(**cfg.executor_cfg)
    webcam_exe.run()


if __name__ == '__main__':
    runWebcamDetection()
