'''This script is for generating data
1. Provide desired path to store images.
2. Press 'c' to capture image and display it.
3. Press any button to continue.
4. Press 'q' to quit.
'''

import cv2
from pathlib import Path

camera = cv2.VideoCapture(0)

print('focus',camera.get(cv2.CAP_PROP_FOCUS))
camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
#camera.set(cv2.CAP_PROP_FOCUS, -3)

print('focus',camera.get(cv2.CAP_PROP_FOCUS))
print('width',camera.get(cv2.CAP_PROP_FRAME_WIDTH ))
#camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960 )
#camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720 )
print('width',camera.get(cv2.CAP_PROP_FRAME_HEIGHT ))
#ret, img = camera.read()


root = Path(__file__).parent.absolute()
path = root.joinpath("aruco_data_new")
#path = "/home/abhishek/stuff/object_detection/explore/aruco_data_new/"
print("PATH", path)

count = 0
while True:
    name = str(path) + "/" + str(count)+".jpg"
    ret, img = camera.read()
    cv2.imshow("img", img)

    if cv2.waitKey(20) & 0xFF == ord('c'):
        cv2.imwrite(name, img)
        cv2.imshow("img", img)
        count += 1
        if cv2.waitKey(0) & 0xFF == ord('q'):

            break;
