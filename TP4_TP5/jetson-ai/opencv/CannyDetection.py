import sys
import argparse
import subprocess

import cv2
import numpy as np

WINDOW_NAME = 'CannyDemo'


def parse_args():
    # Parse input arguments
    desc = 'Capture and display live camera video on Jetson TX2/TX1'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--width', dest='image_width',
                        help='image width [1280]',
                        default=1280, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [720]',
                        default=720, type=int)
    args = parser.parse_args()
    return args


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


def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Canny Edge Detection')


def read_cam(cap):
    show_help = True
    font = cv2.FONT_HERSHEY_PLAIN
    showWindow = 3
    helpText = "'Esc' to Quit, '1' for Camera Feed, '2' for Canny Detection, '3' for All Stages. '4' to hide help, 'd/i' decrease/increase canny edge threshold"
    edgeThreshold = 40

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # Check to see if the user has closed the window
            # If yes, terminate the program
            break
        _, frame = cap.read() # grab the next image frame from camera
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(hsv,(7,7),1.5)
        edges = cv2.Canny(blur,0,edgeThreshold)

        if showWindow == 3:  # Need to show the 4 stages
          # Compose a 2x2 window
          # Stream from the camera is RGB, the others gray
          # To compose, convert gray images to color.
          # All images must be of the same type to display in a window
          frameRs = cv2.resize(frame, (640,480))
          hsvRs = cv2.resize(hsv,(640,480))
          vidBuf = np.concatenate((frameRs, cv2.cvtColor(hsvRs,cv2.COLOR_GRAY2BGR)), axis=1)
          blurRs = cv2.resize(blur,(640,480))
          edgesRs = cv2.resize(edges,(640,480))
          vidBuf1 = np.concatenate( (cv2.cvtColor(blurRs,cv2.COLOR_GRAY2BGR),cv2.cvtColor(edgesRs,cv2.COLOR_GRAY2BGR)), axis=1)
          vidBuf = np.concatenate( (vidBuf, vidBuf1), axis=0)

        if showWindow == 1: # Show Camera Frame
          displayBuf = frame
        elif showWindow == 2: # Show Canny Edge Detection
          displayBuf = edges
        elif showWindow == 3: # Show All Stages
          displayBuf = vidBuf

        if show_help == True:
          cv2.putText(displayBuf, helpText, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
          cv2.putText(displayBuf, helpText, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME,displayBuf)

        key = cv2.waitKey(10)
        if key == 27: # ESC key: quit program
            break
        elif key == ord('H') or key == ord('h'): # toggle help message
            show_help = not show_help
        elif key == ord('1'): # 1 key, show frame
            cv2.setWindowTitle(WINDOW_NAME,"Camera Feed")
            showWindow = 1
        elif key == ord('2'): # 2 key, show Canny
          cv2.setWindowTitle(WINDOW_NAME,"Canny Edge Detection")
          showWindow = 2
        elif key == ord('3'): # 3 key, show Stages
          cv2.setWindowTitle(WINDOW_NAME,"Camera, Gray scale, Gaussian Blur, Canny Edge Detection")
          showWindow = 3
        elif key == ord('4'): # 4 key, toggle help
          show_help = not show_help
        elif key == ord('D') or key == ord('d'): # decrease canny edge threshold
          edgeThreshold = max(0, edgeThreshold-1)
          print('Canny Edge Threshold : ', edgeThreshold)
        elif key == ord('I') or key == ord('i'): # increase canny edge threshold
          edgeThreshold = edgeThreshold + 1
          print('Canny Edge Threshold : ', edgeThreshold)


def main():
    args = parse_args()
    print('Called with args:')
    print(args)
    print('OpenCV version: {}'.format(cv2.__version__))

    # use the Jetson onboard camera
    cap = open_cam_onboard(args.image_width,
                           args.image_height)

    if not cap.isOpened():
        sys.exit('Failed to open camera!')

    open_window(args.image_width, args.image_height)
    read_cam(cap)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()