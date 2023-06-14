#SETTING UP THE ENVIRONMENT

conda create -n newTFM_env python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate newTFM_env

#enter the folder desired

#HUMAN DETECTION CODE

pip3 install openmim
mim install mmcv-full

git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip3 install -e .
mim download mmpose --config associative_embedding_hrnet_w32_coco_512x512  --dest .
mim download mmpose --config associative_embedding_hrnet_w32_coco_512x512  --dest .
cd ..

python -m pip install --upgrade pip
pip install -r requirements.txt

conda install pytorch torchvision cpuonly -c pytorch	#already installed #this is for cpu, in case of usin gpu use: conda install pytorch torchvision -c pytorch
pip install open3d>=0.16*	#already installed
pip install chardet	#already installed

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
mim download mmdet --config faster_rcnn_r50_fpn_1x_coco --dest .
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .	#la han cambiado??
cd ..

#OPENCV (contrib)
sudo apt update && sudo apt install -y cmake g++ wget unzip
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip
mkdir -p build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
cmake --build .
