from __future__ import print_function
from matplotlib import pyplot as plt
from common import splitfn
from videocaptureasync import VideoCaptureAsync
import cv2
import time
import numpy as np
import glob
import os

camera_matrix = np.array([[464.79866571, 0.00000000e+00, 390.09126831],
                          [0.00000000e+00, 401.70010401, 238.70432771],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]);
dist_coefs = np.array([-4.08457447e-01,  1.90178632e-01, 3.09932708e-04, 3.92315248e-04, -4.54584300e-02]);

def test(n_frames=500, width=1280, height=720):

	cap1 = VideoCaptureAsync("rtsp://admin:12345qazqaz@192.168.15.200:554/moxa-cgi/udpstream_ch1")
	cap2 = VideoCaptureAsync("rtsp://admin:12345qazqaz@192.168.15.200:554/moxa-cgi/udpstream_ch2")
	cap3 = VideoCaptureAsync("rtsp://admin:12345qazqaz@192.168.15.200:554/moxa-cgi/udpstream_ch3")
	cap4 = VideoCaptureAsync("rtsp://admin:12345qazqaz@192.168.15.200:554/moxa-cgi/udpstream_ch4")
	cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	cap3.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	cap4.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	cap4.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (720, 480), 1, (720, 480)) 
	x, y, w, h = roi
	#M = cv2.getRotationMatrix2D((720,480),5,2)

        cap1.start()
	cap2.start()
	cap3.start()
	cap4.start()
    	while 1:
        	_, frame1 = cap1.read()
		_, frame2 = cap2.read()
        	_, frame3 = cap3.read()
        	_, frame4 = cap4.read()
        	frame11 = cv2.undistort(frame1, camera_matrix, dist_coefs, None, newcameramtx)
    		#frame11 = frame11[130:390, 110:550]
		frame22 = cv2.undistort(frame2, camera_matrix, dist_coefs, None, newcameramtx)
    		#frame22 = frame22[130:390, 110:550]
		frame33 = cv2.undistort(frame3, camera_matrix, dist_coefs, None, newcameramtx)
    		#frame33 = frame33[130:390, 110:550]
		frame44 = cv2.undistort(frame4, camera_matrix, dist_coefs, None, newcameramtx)
    		#frame44 = frame44[130:390, 110:550]
        	cv2.imshow('Frame1', frame1)
		cv2.imshow('Frame11', frame11)
        	cv2.imshow('Frame2', frame2)
		cv2.imshow('Frame22', frame22)
        	cv2.imshow('Frame3', frame3)
		cv2.imshow('Frame33', frame33)
        	cv2.imshow('Frame4', frame4)
		cv2.imshow('Frame44', frame44)

		if cv2.waitKey(1)==27:
			cv2.destroyAllWindows()
			break

        cap1.stop()
	cap2.stop()
	cap3.stop()
	cap4.stop()
	cv2.destroyAllWindows()

if __name__ == '__main__':
    test(n_frames=5000, width=720, height=480)
