from __future__ import print_function
from matplotlib import pyplot as plt
from common import splitfn
from videocaptureasync import VideoCaptureAsync
import cv2
import time
import numpy as np, cv2
import glob
import argparse
import os
import math

DIM=(720, 480)
K=np.array([[465.3772352680439, 0.0, 391.2362571961526], [0.0, 402.25439297954256, 239.0123153320962], [0.0, 0.0, 1.0]])
D=np.array([[-0.10200056535564077], [0.0027494556950391534], [-0.021553985807500623], [0.011534466820150534]])

def overlap_mbk(a, b):
    a1=np.argsort(a)
    b1=np.argsort(b)
    # use searchsorted:
    sort_left_a=a[a1].searchsorted(b[b1], side='left')
    sort_right_a=a[a1].searchsorted(b[b1], side='right')
    #
    sort_left_b=b[b1].searchsorted(a[a1], side='left')
    sort_right_b=b[b1].searchsorted(a[a1], side='right')


    # # which values are in b but not in a?
    # inds_b=(sort_right_a-sort_left_a == 0).nonzero()[0]
    # # which values are in b but not in a?
    # inds_a=(sort_right_b-sort_left_b == 0).nonzero()[0]

    # which values of b are also in a?
    inds_b=(sort_right_a-sort_left_a > 0).nonzero()[0]
    # which values of a are also in b?
    inds_a=(sort_right_b-sort_left_b > 0).nonzero()[0]

    return a1[inds_a], b1[inds_b]

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))

def undistort(img):
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def nothing(x):
    pass
	
def get_transformation(w, h, alpha, beta, gamma, dist, focalLength):
	A1 = np.array([
		[1, 0, -w/2],
		[0, 1, -h/2],
		[0, 0, 0],
		[0, 0, 1]])

	RX = np.array([
		[1, 0, 0, 0],
		[0, math.cos(alpha), -math.sin(alpha), 0],
		[0, math.sin(alpha), math.cos(alpha), 0],
		[0, 0, 0, 1]])
	
	RY = np.array([
		[math.cos(beta), 0, -math.sin(beta), 0],
		[0, 1, 0, 0],
		[math.sin(beta), 0, math.cos(beta), 0],
		[0, 0, 0, 1]])

	RZ = np.array([
		[math.cos(gamma), -math.sin(gamma), 0, 0],
		[math.sin(gamma), math.cos(gamma), 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]])

	R = np.dot(RY, RZ)
	R = np.dot(R, RX)

	T = np.array([
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, dist],
		[0, 0, 0, 1]])

	K = np.array([
		[focalLength, 0, w/2, 0],
		[0, focalLength, h/2, 0],
		[0, 0, 1, 0]])
	
	w1 = np.dot(R, A1)
	w2 = np.dot(T, w1)
	transformation = np.dot(K, w2)

	#print("alpha: ", alpha)
	#print("cos: ", math.cos(alpha))
	#print("RX: ", RX)
	#print("RY: ", RY)
	#print("RZ: ", RZ)
	#print("R: ", R)
	#print("A1: ", A1)
	#print("T: ", T)
	#print("K: ", K)
	#print("Mat: ", transformation)	
	
	return transformation

def test(width, height):

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
	
	alpha = 90
	beta = 90
	gamma = 90
	focalLength = 500
	dist = 500
	move_upper = 0
	move_left = 0
	move_right = 0
	move_lower = 0
	move_together = 0
	cv2.namedWindow('tune_upper')
	cv2.namedWindow('tune_left')
	cv2.namedWindow('tune_right')
	cv2.namedWindow('tune_lower')
	cv2.namedWindow('move')

	cv2.createTrackbar('Alpha','tune_upper',60,180,nothing)
	cv2.createTrackbar('Beta','tune_upper',90,180,nothing)
	cv2.createTrackbar('Gamma','tune_upper',90,180,nothing)
	cv2.createTrackbar('f','tune_upper',300,2000,nothing)
	cv2.createTrackbar('Distance','tune_upper',500,2000,nothing)

	cv2.createTrackbar('Alpha','tune_left',90,180,nothing)
	cv2.createTrackbar('Beta','tune_left',70,180,nothing)
	cv2.createTrackbar('Gamma','tune_left',90,180,nothing)
	cv2.createTrackbar('f','tune_left',300,2000,nothing)
	cv2.createTrackbar('Distance','tune_left',500,2000,nothing)
	
	cv2.createTrackbar('Alpha','tune_right',90,180,nothing)
	cv2.createTrackbar('Beta','tune_right',110,180,nothing)
	cv2.createTrackbar('Gamma','tune_right',90,180,nothing)
	cv2.createTrackbar('f','tune_right',300,2000,nothing)
	cv2.createTrackbar('Distance','tune_right',500,2000,nothing)

	cv2.createTrackbar('Alpha','tune_lower',60,180,nothing)
	cv2.createTrackbar('Beta','tune_lower',90,180,nothing)
	cv2.createTrackbar('Gamma','tune_lower',90,180,nothing)
	cv2.createTrackbar('f','tune_lower',300,2000,nothing)
	cv2.createTrackbar('Distance','tune_lower',500,2000,nothing)

	cv2.createTrackbar('Move upper','move',0,400,nothing)
	cv2.createTrackbar('Move left','move',0,400,nothing)
	cv2.createTrackbar('Move right','move',0,400,nothing)
	cv2.createTrackbar('Move lower','move',0,400,nothing)
	cv2.createTrackbar('Move together','move',200,400,nothing)


        cap1.start()
	cap2.start()
	cap3.start()
	cap4.start()
	h = 480
	w = 720	
	vis = np.zeros(((w+h*2),(w+h*2),3),np.uint8)
	vis_center = np.zeros(((w+h*2),h,3),np.uint8)
	vis_left = np.zeros(((w+h*2),h,3),np.uint8)
	vis_right = np.zeros(((w+h*2),h,3),np.uint8)
	void = np.zeros((w,w,3),np.uint8)
	void_side = np.zeros((h,h,3),np.uint8)

	
	

    	while 1:
        	_, frame1 = cap1.read()
		_, frame2 = cap2.read()
        	_, frame3 = cap3.read()
        	_, frame4 = cap4.read()
		
        	frame11 = undistort(frame1)
		frame22 = undistort(frame2)
		frame33 = undistort(frame3)
		frame44 = undistort(frame4)

		alpha_upper = cv2.getTrackbarPos('Alpha','tune_upper')
		beta_upper = cv2.getTrackbarPos('Beta','tune_upper')
		gamma_upper = cv2.getTrackbarPos('Gamma','tune_upper')
		focalLength_upper = cv2.getTrackbarPos('f','tune_upper')
		dist_upper = cv2.getTrackbarPos('Distance','tune_upper')

		alpha_left = cv2.getTrackbarPos('Alpha','tune_left')
		beta_left = cv2.getTrackbarPos('Beta','tune_left')
		gamma_left = cv2.getTrackbarPos('Gamma','tune_left')
		focalLength_left = cv2.getTrackbarPos('f','tune_left')
		dist_left = cv2.getTrackbarPos('Distance','tune_left')

		alpha_right = cv2.getTrackbarPos('Alpha','tune_right')
		beta_right = cv2.getTrackbarPos('Beta','tune_right')
		gamma_right = cv2.getTrackbarPos('Gamma','tune_right')
		focalLength_right = cv2.getTrackbarPos('f','tune_right')
		dist_right = cv2.getTrackbarPos('Distance','tune_right')

		alpha_lower = cv2.getTrackbarPos('Alpha','tune_lower')
		beta_lower = cv2.getTrackbarPos('Beta','tune_lower')
		gamma_lower = cv2.getTrackbarPos('Gamma','tune_lower')
		focalLength_lower = cv2.getTrackbarPos('f','tune_lower')
		dist_lower = cv2.getTrackbarPos('Distance','tune_lower')

		move_upper = cv2.getTrackbarPos('Move upper','move')
		move_left = cv2.getTrackbarPos('Move left','move')
		move_right = cv2.getTrackbarPos('Move right','move')
		move_lower = cv2.getTrackbarPos('Move lower','move')
		move_together = cv2.getTrackbarPos('Move together','move')
		
		alpha_upper = (alpha_upper - 90) * math.pi/180
		beta_upper = (beta_upper - 90) * math.pi/180
		gamma_upper = (gamma_upper - 90) * math.pi/180

		alpha_left = (alpha_left - 90) * math.pi/180
		beta_left = (beta_left - 90) * math.pi/180
		gamma_left = (gamma_left - 90) * math.pi/180

		alpha_right = (alpha_right - 90) * math.pi/180
		beta_right = (beta_right - 90) * math.pi/180
		gamma_right = (gamma_right - 90) * math.pi/180

		alpha_lower = (alpha_lower - 90) * math.pi/180
		beta_lower = (beta_lower - 90) * math.pi/180
		gamma_lower = (gamma_lower - 90) * math.pi/180
		
		#rotation_mat = cv2.getRotationMatrix2D((360,240), 180, 1)
		#print("Transformation: ", Transformation.shape)
		#print("Rotation: ", rotation_mat.shape)
		
		transformation_upper=get_transformation(w,h,alpha_upper,beta_upper,gamma_upper,dist_upper,focalLength_upper)
		transformation_left=get_transformation(h,w,alpha_left,beta_left,gamma_left,dist_left,focalLength_left)
		transformation_right=get_transformation(h,w,alpha_right,beta_right,gamma_right,dist_right,focalLength_right)
		transformation_lower=get_transformation(w,h,alpha_lower,beta_lower,gamma_lower,dist_lower,focalLength_lower)
		
		left = rotate_bound(frame33, 270)
		right = rotate_bound(frame22, 90)

		result_upper = cv2.warpPerspective(frame11, transformation_upper, (720, 480))
		result_left = cv2.warpPerspective(left, transformation_left, (480, 720))
		result_right = cv2.warpPerspective(right, transformation_right, (480, 720))
		result_lower = cv2.warpPerspective(frame44, transformation_lower, (720, 480))

        	#cv2.imshow('Frame1', frame1)
		#cv2.imshow('Frame11', frame11)
        	#cv2.imshow('Frame2', frame2)
		#cv2.imshow('Frame22', frame22)
        	#cv2.imshow('Frame3', frame3)
		#cv2.imshow('Frame33', frame33)
        	#cv2.imshow('Frame4', frame4)
		#cv2.imshow('Frame44', frame44)
		#left = rotate_bound(result_left, 270)
		#right = rotate_bound(result_right, 90)
		result_lower = cv2.flip(result_lower, 0)
		#cv2.imshow('left', left)
		#cv2.imshow('right',right)		
		#vis_center = np.concatenate((result,void,frame44),axis=0)
		#vis_right = np.concatenate((void_side,right,void_side),axis=0)
		#vis_left = np.concatenate((void_side,left,void_side),axis=0)
		#vis = np.concatenate((vis_left,vis_center,vis_right),axis=1)
		vis[move_upper + move_together:result_upper.shape[0] + move_upper + move_together, h-(result_upper.shape[1]-720)/2:result_upper.shape[1] + h-(result_upper.shape[1]-720)/2,:] = result_upper
		vis[h:result_left.shape[0]+h,move_left + move_together:result_left.shape[1] + move_left + move_together,:] += result_left
		vis[h:result_right.shape[0]+h,h + w - move_right - move_together:result_right.shape[1] - move_right - move_together + w + h,:] += result_right
		vis[h+w-move_lower - move_together:result_lower.shape[0] - move_lower - move_together+w+h, h:result_lower.shape[1] + h,:] += result_lower
		height, width = vis.shape[:2]
		res = cv2.resize(vis,(width*4/7, height*4/7), interpolation = cv2.INTER_CUBIC)
		#cv2.imshow('combined', vis_center)
		#cv2.imshow('combined_left', vis_left)
		#cv2.imshow('combined_right', vis_right)
		cv2.imshow('vis', res)		
		vis = np.zeros(((w+h*2),(w+h*2),3),np.uint8)
		if cv2.waitKey(1)==27:
			cv2.destroyAllWindows()
			break

        cap1.stop()
	cap2.stop()
	cap3.stop()
	cap4.stop()
	cv2.destroyAllWindows()

if __name__ == '__main__':
    test(width=720, height=480)
