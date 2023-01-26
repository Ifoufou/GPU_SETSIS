# https://devtalk.nvidia.com/default/topic/1027250/jetson-tx2/how-to-use-usb-webcam-in-jetson-tx2-with-python-and-opencv-/
# To run the program, type
#   python3 mnist-camera.py
# Type 'q' to quit

from __future__ import print_function
from keras.models import load_model
import cv2
import numpy as np

import sys
import argparse
import subprocess

# built by copy-pasting snippets from tegra-cam.py and mnist_console.py
def open_cam_onboard(width, height):
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

# Load CNN model, assuming that you save your CNN in mnist/models/mnist_cnn.h5
model_file = 'models/mnist_cnn.h5'
print('Loading %s' % model_file)
model = load_model(model_file)

kernel = np.ones((5,5), np.uint8)

#cap = cv2.VideoCapture("/dev/video1")       # video0 is the built-in cam and video1 is the webcam
# use the Jetson onboard camera
cap = open_cam_onboard(640,
                       480)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()                 # frame is a numpy array with shape (480, 640, 3)
    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # frame is a numpy array with shape (480, 640)
    # Increase the contrast to make match the training data
    frame = cv2.addWeighted(frame, 2, frame, 0, 10)
    # Show the image on screen
    cv2.imshow('frame', frame)
    # dilatation
    frame = cv2.erode(frame, kernel, iterations=15)
    # Resize to (28,28)
    img = cv2.resize(frame, (28, 28))
    cv2.imshow('resized', img)
    
    # Use black background and scale to [0,1] to match the training data
    img = (1-img)/255              
    cv2.imshow('toto', img)

    # Reshape the image from (28,28) to (1,28,28,1) to fit the input of your CNN
    # Put your code here. Search for "Numpy reshape"
    x = np.reshape(img, [1,28,28,1])

    # Present the image to CNN for classification
    # Put your code here. Search for "Keras model predict_class" from the Internet.
    class_lb = model.predict_classes(x)
    print(class_lb)

    k = cv2.waitKey(10) & 0xFF
    if k == ord('q'):
        break
            
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
