from __future__ import print_function
from matplotlib import pyplot as plt
from common import splitfn
from videocaptureasync import VideoCaptureAsync
import cv2
import glob
import os

def test(width=1280, height=720):

	#cap1 = VideoCaptureAsync("rtsp://admin:12345qazqaz@192.168.15.200:554/moxa-cgi/udpstream_ch1")
	#cap1 = VideoCaptureAsync('udpsrc ! application/x-rtp, encoding-name=JPEG, payload=26 ! rtpjpegdepay ! jpegdec ! autovideosink port=50004', cv2.CAP_GSTREAMER)
	cap1 = cv2.VideoCapture('udpsrc port=50004 caps = "application/x-rtp,media=video,payload=26,clock-rate=90000,encoding-name=JPEG,framerate=30/1" ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink')
	#cap2 = VideoCaptureAsync("rtsp://admin:12345qazqaz@192.168.15.200:554/moxa-cgi/udpstream_ch2")
	#cap3 = VideoCaptureAsync("rtsp://admin:12345qazqaz@192.168.15.200:554/moxa-cgi/udpstream_ch3")
	#cap4 = VideoCaptureAsync("rtsp://admin:12345qazqaz@192.168.15.200:554/moxa-cgi/udpstream_ch4")

        #cap1.start()
	#cap2.start()
	#cap3.start()
	#cap4.start()
	i=0
    	while 1:
        	ret, frame1 = cap1.read()
		#_, frame2 = cap2.read()
        	#_, frame3 = cap3.read()
        	#_, frame4 = cap4.read()
		#print(cv2.getBuildInformation())
		print(cap1.isOpened())
		#if not ret:
            	#	print('empty frame')
            	#	break

        	cv2.imshow('Frame1', frame1)
        	#cv2.imshow('Frame2', frame2)
        	#cv2.imshow('Frame3', frame3)
        	#cv2.imshow('Frame4', frame4)

        	if cv2.waitKey(1)==27:
			cv2.destroyAllWindows()
			break
		elif cv2.waitKey(20)==ord('s'):
			cv2.imwrite('undistor/cam1_{:>02}.png'.format(i), frame1)
			#cv2.imwrite('undistor/cam2_{:>02}.png'.format(i), frame2)
			#cv2.imwrite('undistor/cam3_{:>02}.png'.format(i), frame3)
			#cv2.imwrite('undistor/cam4_{:>02}.png'.format(i), frame4)
			i=i+1

        #cap1.stop()
	#cap2.stop()
	#cap3.stop()
	#cap4.stop()
	cv2.destroyAllWindows()

if __name__ == '__main__':
    test(width=1280, height=720)
