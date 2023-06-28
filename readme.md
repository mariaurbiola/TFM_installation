# SETTING UP THE ENVIRONMENT
#enter the folder where you want your work to be, in my case TFM_MariaUrbiola

conda create -n TFM_env python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y

conda activate TFM_env

python -m pip install --upgrade pip

pip install -r requirements.txt


# HUMAN DETECTION CODE

# Instalation of the repositories for detecting people
pip3 install openmim

mim install mmcv-full


cd mmpose

pip3 install -e .

cd ..



git clone https://github.com/open-mmlab/mmdetection.git

cd mmdetection

pip install -r requirements/build.txt

pip install -v -e .

cd ..



# OPENCV
1. pip install opencv-contrib-python==4.6.0.66

