import cv2
import time
from videocaptureasync import VideoCaptureAsync

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

        cap1.start()
	cap2.start()
	cap3.start()
	cap4.start()
    	while 1:
        	_, frame1 = cap1.read()
		_, frame2 = cap2.read()
        	_, frame3 = cap3.read()
        	_, frame4 = cap4.read()
        
        	cv2.imshow('Frame1', frame1)
        	cv2.imshow('Frame2', frame2)
        	cv2.imshow('Frame3', frame3)
        	cv2.imshow('Frame4', frame4)


        	cv2.waitKey(1) & 0xFF

        cap1.stop()
	cap2.stop()
	cap3.stop()
	cap4.stop()
	cv2.destroyAllWindows()

if __name__ == '__main__':
    test(n_frames=5000, width=720, height=640)
