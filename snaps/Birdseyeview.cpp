
#include <opencv2/opencv.hpp>
#include <iostream>
#include <bits/stdc++.h>
#include <chrono>
#include "opencv2/stitching/detail/exposure_compensate.hpp"


#define PI 3.1415926

using namespace std;
using namespace cv;
using namespace cv::detail;

Mat getTransformation(int w, int h, int alpha_, int beta_, int gamma_, int f_, int dist_){

		double alpha =((double)alpha_ -90) * PI/180;
		double beta =((double)beta_ -90) * PI/180;
		double gamma =((double)gamma_ -90) * PI/180;
		double focalLength = (double)f_;
		double dist = (double)dist_;

		// Projecion matrix 2D -> 3D
		Mat A1 = (Mat_<float>(4, 3)<< 
			1, 0, -w/2,
			0, 1, -h/2,
			0, 0, 0,
			0, 0, 1 );
	
		// Rotation matrices Rx, Ry, Rz

		Mat RX = (Mat_<float>(4, 4) << 
			1, 0, 0, 0,
			0, cos(alpha), -sin(alpha), 0,
			0, sin(alpha), cos(alpha), 0,
			0, 0, 0, 1 );

		Mat RY = (Mat_<float>(4, 4) << 
			cos(beta), 0, -sin(beta), 0,
			0, 1, 0, 0,
			sin(beta), 0, cos(beta), 0,
			0, 0, 0, 1	);

		Mat RZ = (Mat_<float>(4, 4) << 
			cos(gamma), -sin(gamma), 0, 0,
			sin(gamma), cos(gamma), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1	);

		// R - rotation matrix
		Mat R = RX * RY * RZ;

		// T - translation matrix
		Mat T = (Mat_<float>(4, 4) << 
			1, 0, 0, 0,  
			0, 1, 0, 0,  
			0, 0, 1, dist,  
			0, 0, 0, 1); 
		
		// K - intrinsic matrix 
		Mat K = (Mat_<float>(3, 4) << 
			focalLength, 0, w/2, 0,
			0, focalLength, h/2, 0,
			0, 0, 1, 0
			); 

		Mat transformationMat = K * (T * (R * A1));
		
	return transformationMat;
}

Mat rotateBound(Mat src, int angle){

	// get rotation matrix for rotating the image around its center in pixel coordinates
	Point2f center((src.cols-1)/2.0, (src.rows-1)/2.0);
	Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
	// determine bounding rectangle, center not relevant
	Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
	// adjust transformation matrix
	rot.at<double>(0,2) += bbox.width/2.0 - src.cols/2.0;
	rot.at<double>(1,2) += bbox.height/2.0 - src.rows/2.0;

	Mat dst;
	warpAffine(src, dst, rot, bbox.size());

	return dst;
}

// Main body

int main()
{	
	//cout<<getBuildInformation()<<endl;
	int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
	vector<Point> corners(4);
	vector<UMat> images(4);
	vector<UMat> masks(4);

	Mat compensated_front, compensated_right, compensated_rear, compensated_left;

	setUseOptimized(true);
	Mat cameraMatrix = (Mat_<double>(3,3) << 4.3526298745939096e+02, 0., 6.3950000000000000e+02, 0., 4.3415306567637083e+02, 3.5950000000000000e+02, 0., 0., 1.);
	Mat test = (Mat_<double>(3,3) << 4, 0, 8, 0, 4, 6, 0, 0, 1);
	Mat distCoeffs = (Mat_<double>(4,1) << -4.5133663311739985e-02, 2.1055083480493644e-02,
       -1.0118009701988350e-02, 0.);
	//   FRONT
	Mat cameraMatrixFront = (Mat_<double>(3,3) << 3.7299931648784468e+02, 0., 6.3950000000000000e+02, 0., 3.6025039442103241e+02, 3.5950000000000000e+02, 0., 0., 1.);
	Mat distCoeffsFront = (Mat_<double>(4,1) << -3.9258100334454066e-02, 1.2233971618735032e-02,
       -6.1130900000061564e-03, 0.);
	//   RIGHT
	Mat cameraMatrixRight = (Mat_<double>(3,3) << 4.4290410995281678e+02, 0., 6.3950000000000000e+02, 0., 4.4457276594761350e+02, 3.5950000000000000e+02, 0., 0., 1.);
	Mat distCoeffsRight = (Mat_<double>(4,1) << -2.2481663392985859e-01, 2.3213490292261166e-01,
      -7.2733393065457452e-02, 0.);
	//   LEFT
	Mat cameraMatrixLeft = (Mat_<double>(3,3) << 3.9804643129477353e+02, 0., 6.3950000000000000e+02, 0., 3.9163209989364623e+02, 3.5950000000000000e+02, 0., 0., 1.);
	Mat distCoeffsLeft = (Mat_<double>(4,1) << -1.5635956575058771e-01, 1.7610712797773820e-01,
       -5.1546134685890602e-02, 0.);
	//   REAR
	Mat cameraMatrixRear = (Mat_<double>(3,3) << 3.4690415081253752e+02, 0., 6.3950000000000000e+02, 0., 3.6257675262825472e+02, 3.5950000000000000e+02, 0., 0., 1.);
	Mat distCoeffsRear = (Mat_<double>(4,1) << -9.3059362634166972e-03, 3.6386888022608359e-02,
       -8.8726768045285127e-03, 0.);

	Mat frame_left(Size(1280,720), CV_32FC3);
	Mat frame_right(Size(1280,720), CV_32FC3);
	Mat frame_front(Size(1280,720), CV_32FC3);
	Mat frame_rear(Size(1280,720), CV_32FC3);
	Mat view, newCamMat, newCamMatFront ,newCamMatRear ,newCamMatLeft ,newCamMatRight;
	Mat undist_left(Size(1280,720),CV_32FC3);
	Mat undist_right(Size(1280,720),CV_32FC3);
	Mat undist_front(Size(1280,720),CV_32FC3);
	Mat undist_rear(Size(1280,720),CV_32FC3);
	Mat map1(Size(1280,720),CV_16SC2);
	Mat map2(Size(1280,720),CV_16UC1);
	Mat map1Front(Size(1280,720),CV_16SC2);
	Mat map2Front(Size(1280,720),CV_16UC1);
	Mat map1FrontDst(Size(1280,720),CV_16SC2);
	Mat map2FrontDst(Size(1280,720),CV_16UC1);
	Mat map1Rear(Size(1280,720),CV_16SC2);
	Mat map2Rear(Size(1280,720),CV_16UC1);
	Mat map1RearDst(Size(1280,720),CV_16SC2);
	Mat map2RearDst(Size(1280,720),CV_16UC1);
	Mat map1Right(Size(1280,720),CV_16SC2);
	Mat map2Right(Size(1280,720),CV_16UC1);
	Mat map1RightDst(Size(1280,720),CV_16SC2);
	Mat map2RightDst(Size(1280,720),CV_16UC1);
	Mat map1Left(Size(1280,720),CV_16SC2);
	Mat map2Left(Size(1280,720),CV_16UC1);
	Mat map1LeftDst(Size(1280,720),CV_16SC2);
	Mat map2LeftDst(Size(1280,720),CV_16UC1);
	Mat front = Mat::zeros(Size(1000,1500), CV_32FC3);
	Mat rear = Mat::zeros(Size(1000,1500), CV_32FC3);
	Mat left = Mat::zeros(Size(1000,1500), CV_32FC3);
	Mat right = Mat::zeros(Size(1000,1500), CV_32FC3);
	int h = 720;
	int w = 1280;
	Mat vis = Mat::zeros(Size((w+h*2),(w+h*2)),CV_32FC3);
	Mat vis_front = Mat::zeros(Size((w+h*2),(w+h*2)),CV_32FC3);
	Mat vis_left = Mat::zeros(Size((w+h*2),(w+h*2)),CV_32FC3);
	Mat vis_right = Mat::zeros(Size((w+h*2),(w+h*2)),CV_32FC3);
	Mat vis_rear = Mat::zeros(Size((w+h*2),(w+h*2)),CV_32FC3);

	VideoCapture cap_right("udpsrc port=50003 ! application/x-rtp,media=video,payload=26,encoding-name=JPEG,framerate=30/1 ! queue ! rtpjpegdepay ! jpegdec ! video/x-raw,format=I420 ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink", CAP_GSTREAMER);
    	VideoCapture cap_rear("udpsrc port=50004 ! application/x-rtp,media=video,payload=26,encoding-name=JPEG,framerate=30/1 ! queue ! rtpjpegdepay ! jpegdec ! video/x-raw,format=I420 ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink", CAP_GSTREAMER);
	VideoCapture cap_front("udpsrc port=50005 ! application/x-rtp,media=video,payload=26,encoding-name=JPEG,framerate=30/1 ! queue ! rtpjpegdepay ! jpegdec ! video/x-raw,format=I420 ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink", CAP_GSTREAMER);
	VideoCapture cap_left("udpsrc port=50006 ! application/x-rtp,media=video,payload=26,encoding-name=JPEG,framerate=30/1 ! queue ! rtpjpegdepay ! jpegdec ! video/x-raw,format=I420 ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink", CAP_GSTREAMER);

	/*VideoCapture cap1("/home/aleksandr/birds_eye_view_system/test1.mp4");
	VideoCapture cap_rear("/home/aleksandr/birds_eye_view_system/test2.mp4");
	VideoCapture cap_front("/home/aleksandr/birds_eye_view_system/test3.mp4");
	VideoCapture cap_left("/home/aleksandr/birds_eye_view_system/test4.mp4");*/	

	if (!cap_right.isOpened()) {
        cerr <<"VideoCapture right not opened"<<endl;
        exit(-1);
    }
	/*if (!cap_rear.isOpened()) {
        cerr <<"VideoCapture rear not opened"<<endl;
        exit(-1);
    }
	if (!cap_front.isOpened()) {
        cerr <<"VideoCapture front not opened"<<endl;
        exit(-1);
    }
	if (!cap_left.isOpened()) {
        cerr <<"VideoCapture left not opened"<<endl;
        exit(-1);
    }*/
    	cap_right.read(frame_right);
	Size imageSize = frame_right.size();
	fisheye::estimateNewCameraMatrixForUndistortRectify(cameraMatrix, distCoeffs, imageSize,Matx33d::eye(), newCamMat, 1);
        fisheye::initUndistortRectifyMap(cameraMatrix, distCoeffs, Matx33d::eye(), newCamMat, imageSize,CV_16SC2, map1, map2);
	Mat map1Dst(map1.size(),CV_16SC2);
	Mat map2Dst(map2.size(),CV_16UC1);
	convertMaps(map1, map2, map1Dst, map2Dst, CV_16SC2);
	// FRONT
	fisheye::estimateNewCameraMatrixForUndistortRectify(cameraMatrixFront, distCoeffsFront, imageSize,Matx33d::eye(), newCamMatFront, 1);
        fisheye::initUndistortRectifyMap(cameraMatrixFront, distCoeffsFront, Matx33d::eye(), newCamMatFront, imageSize,CV_16SC2, map1Front, map2Front);
	convertMaps(map1Front, map2Front, map1FrontDst, map2FrontDst, CV_16SC2);
	
	// REAR
	fisheye::estimateNewCameraMatrixForUndistortRectify(cameraMatrixRear, distCoeffsRear, imageSize,Matx33d::eye(), newCamMatRear, 1);
        fisheye::initUndistortRectifyMap(cameraMatrixRear, distCoeffsRear, Matx33d::eye(), newCamMatRear, imageSize,CV_16SC2, map1Rear, map2Rear);
	convertMaps(map1Rear, map2Rear, map1RearDst, map2RearDst, CV_16SC2);

	// LEFT
	fisheye::estimateNewCameraMatrixForUndistortRectify(cameraMatrixLeft, distCoeffsLeft, imageSize,Matx33d::eye(), newCamMatLeft, 1);
        fisheye::initUndistortRectifyMap(cameraMatrixLeft, distCoeffsLeft, Matx33d::eye(), newCamMatLeft, imageSize,CV_16SC2, map1Left, map2Left);
	convertMaps(map1Left, map2Left, map1LeftDst, map2LeftDst, CV_16SC2);

	// RIGHT
	fisheye::estimateNewCameraMatrixForUndistortRectify(cameraMatrixRight, distCoeffsRight, imageSize,Matx33d::eye(), newCamMatRight, 1);
        fisheye::initUndistortRectifyMap(cameraMatrixRight, distCoeffsRight, Matx33d::eye(), newCamMatRight, imageSize,CV_16SC2, map1Right, map2Right);
	convertMaps(map1Right, map2Right, map1RightDst, map2RightDst, CV_16SC2);
	
	Mat destination_right(frame_right.size(), CV_32FC3);
	Mat destination_left(frame_left.size(), CV_32FC3);
	Mat destination_front(frame_front.size(), CV_32FC3);
	Mat destination_rear(frame_rear.size(), CV_32FC3);
	int alpha_front_ = 99, beta_front_ = 90, gamma_front_ = 90, f_front_ = 300, d_front_ = 361;
	int alpha_left_ = 97, beta_left_ = 90, gamma_left_ = 90, f_left_ = 300, d_left_ = 233;
	int alpha_right_ = 100, beta_right_ = 90, gamma_right_ = 90, f_right_ = 300, d_right_ = 435;
	int alpha_rear_ = 113, beta_rear_ = 90, gamma_rear_ = 88, f_rear_ = 300, d_rear_ = 350;
	int move_front_ = 187, move_left_ = 177, move_right_ = 185, move_rear_ = 90, move_together_ = 645;

	namedWindow("tune_front");
	namedWindow("tune_left");
	namedWindow("tune_right");
	namedWindow("tune_rear");
	namedWindow("move");

	createTrackbar("Alpha","tune_front",&alpha_front_,180);
	createTrackbar("Beta","tune_front",&beta_front_,180);
	createTrackbar("Gamma","tune_front",&gamma_front_,180);
	createTrackbar("f","tune_front",&f_front_,2000);
	createTrackbar("Distance","tune_front",&d_front_,2000);

	createTrackbar("Alpha","tune_left",&alpha_left_,180);
	createTrackbar("Beta","tune_left",&beta_left_,180);
	createTrackbar("Gamma","tune_left",&gamma_left_,180);
	createTrackbar("f","tune_left",&f_left_,2000);
	createTrackbar("Distance","tune_left",&d_left_,2000);
	
	createTrackbar("Alpha","tune_right",&alpha_right_,180);
	createTrackbar("Beta","tune_right",&beta_right_,180);
	createTrackbar("Gamma","tune_right",&gamma_right_,180);
	createTrackbar("f","tune_right",&f_right_,2000);
	createTrackbar("Distance","tune_right",&d_right_,2000);

	createTrackbar("Alpha","tune_rear",&alpha_rear_,180);
	createTrackbar("Beta","tune_rear",&beta_rear_,180);
	createTrackbar("Gamma","tune_rear",&gamma_rear_,180);
	createTrackbar("f","tune_rear",&f_rear_,2000);
	createTrackbar("Distance","tune_rear",&d_rear_,2000);

	createTrackbar("Move front","move",&move_front_,400);
	createTrackbar("Move left","move",&move_left_,400);
	createTrackbar("Move right","move",&move_right_,400);
	createTrackbar("Move rear","move",&move_rear_,400);
	createTrackbar("Move together","move",&move_together_,700);
	
	double focalLength, dist, alpha, beta, gamma;
	cout<<"Optimization: "<<useOptimized()<<'\n';
	int counter = 0;
	Mat bg = imread("bg.png");
    while (true) {
	
	auto start_total = chrono::high_resolution_clock::now();
	auto start_cap  = chrono::high_resolution_clock::now();
	#pragma omp parallel num_threads(4)
	{
		#pragma omp sections
		{		
			#pragma omp section
			{
				cap_right.read(frame_right);
			}
			#pragma omp section
			{
				cap_rear.read(frame_rear);
			}
			#pragma omp section
			{
				cap_front.read(frame_front);
			}
			#pragma omp section
			{
				cap_left.read(frame_left);
			}
		}
	}

	counter++;
	if(counter%2 == 1){

	auto end_cap  = chrono::high_resolution_clock::now();
	
	auto start_undist  = chrono::high_resolution_clock::now();
		//cout<<typeid(map1).name()<<"   "<<typeid(map1Dst).name()<<endl;
		remap(frame_right, undist_right, map1RightDst, map2RightDst, INTER_CUBIC, BORDER_CONSTANT);
		remap(frame_rear, undist_rear, map1Dst, map2Dst, INTER_CUBIC, BORDER_CONSTANT);	
		remap(frame_front, undist_front, map1FrontDst, map2FrontDst, INTER_CUBIC, BORDER_CONSTANT);
		remap(frame_left, undist_left, map1LeftDst, map2LeftDst, INTER_CUBIC, BORDER_CONSTANT);
			
	auto end_undist  = chrono::high_resolution_clock::now();
	
	auto start_perspect  = chrono::high_resolution_clock::now();
		
		alpha_front_ = getTrackbarPos("Alpha","tune_front");
		beta_front_ = getTrackbarPos("Beta","tune_front");
		gamma_front_ = getTrackbarPos("Gamma","tune_front");
		f_front_ = getTrackbarPos("f","tune_front");
		d_front_ = getTrackbarPos("Distance","tune_front");

		alpha_left_ = getTrackbarPos("Alpha","tune_left");
		beta_left_ = getTrackbarPos("Beta","tune_left");
		gamma_left_ = getTrackbarPos("Gamma","tune_left");
		f_left_ = getTrackbarPos("f","tune_left");
		d_left_ = getTrackbarPos("Distance","tune_left");

		alpha_right_ = getTrackbarPos("Alpha","tune_right");
		beta_right_ = getTrackbarPos("Beta","tune_right");
		gamma_right_ = getTrackbarPos("Gamma","tune_right");
		f_right_ = getTrackbarPos("f","tune_right");
		d_right_ = getTrackbarPos("Distance","tune_right");

		alpha_rear_ = getTrackbarPos("Alpha","tune_rear");
		beta_rear_ = getTrackbarPos("Beta","tune_rear");
		gamma_rear_ = getTrackbarPos("Gamma","tune_rear");
		f_rear_ = getTrackbarPos("f","tune_rear");
		d_rear_ = getTrackbarPos("Distance","tune_rear");

		move_front_ = getTrackbarPos("Move front","move");
		move_left_ = getTrackbarPos("Move left","move");
		move_right_ = getTrackbarPos("Move right","move");
		move_rear_ = getTrackbarPos("Move rear","move");
		move_together_ = getTrackbarPos("Move together","move");

	
	Mat transform_front = getTransformation(1280, 720, alpha_front_, beta_front_, gamma_front_, f_front_, d_front_);
	warpPerspective(undist_front, destination_front, transform_front, imageSize, INTER_NEAREST );
			
	Mat transform_left = getTransformation(1280, 720, alpha_left_, beta_left_, gamma_left_, f_left_, d_left_);
	warpPerspective(undist_left, destination_left, transform_left, imageSize, INTER_NEAREST );
			
	Mat transform_right = getTransformation(1280, 720, alpha_right_, beta_right_, gamma_right_, f_right_, d_right_);
	warpPerspective(undist_right, destination_right, transform_right, imageSize, INTER_NEAREST );
			
	Mat transform_rear = getTransformation(1280, 720, alpha_rear_, beta_rear_, gamma_rear_, f_rear_, d_rear_);
	warpPerspective(undist_rear, destination_rear, transform_rear, imageSize, INTER_NEAREST );

	imshow("right", destination_right);
	imshow("left", destination_left);
	imshow("front", destination_front);
	imshow("rear", destination_rear);			

	destination_left = rotateBound(destination_left, 270);
	flip(destination_left, destination_left, 1);
	flip(destination_left, destination_left, 0);
	destination_right = rotateBound(destination_right, 90);
	flip(destination_right, destination_right, 1);
	flip(destination_right, destination_right, 0);	
	flip(destination_rear, destination_rear, 0);
	flip(destination_rear, destination_rear, 1);
	
	auto end_perspect  = chrono::high_resolution_clock::now();	
		
	/*masks[0].create(destination_front.size(), CV_8U);
        masks[0].setTo(Scalar::all(255));
	corners[0] = Point(h, 0 + move_together_ + move_front_);
	destination_front.copyTo(images[0]);
	masks[1].create(destination_right.size(), CV_8U);
        masks[1].setTo(Scalar::all(255));
	corners[1] = Point(w + h - (move_together_ + move_right_), h);
	destination_right.copyTo(images[1]);
	masks[2].create(destination_rear.size(), CV_8U);
        masks[2].setTo(Scalar::all(255));
	corners[2] = Point(h, w + h - (move_together_ + move_rear_));
	destination_rear.copyTo(images[2]);
	masks[3].create(destination_left.size(), CV_8U);
        masks[3].setTo(Scalar::all(255));
	corners[3] = Point(0 +  move_together_ + move_left_, h);
	destination_left.copyTo(images[3]);*/

	//Exposure compensator
	//compensator->feed(corners, images, masks);
	/*compensator->apply(0, corners[0], compensated_front, masks[0]);
	compensator->apply(1, corners[1], compensated_right, masks[1]);
	compensator->apply(2, corners[2], compensated_rear, masks[2]);
	compensator->apply(3, corners[3], compensated_left, masks[3]);*/

	auto start_merge  = chrono::high_resolution_clock::now();

	Mat bckg_left = bg;
	Mat bckg_right = bg;
	Mat bckg_front = bg;
	Mat bckg_rear = bg;
	
	/*Mat mask_vert = Mat::zeros(bg.rows, bg.cols, CV_8U);
	Point pts_vert[6] = {
		Point(0, 0),
		Point((h * 2 + w) / 2, (h * 2 + w) / 2),
		Point(0, h * 2 + w),
		Point( h * 2 + w,  h * 2 + w),
		Point((h * 2 + w) / 2, (h * 2 + w) / 2),
		Point(h * 2 + w,0)
	};
	fillConvexPoly( mask_vert, pts_vert, 6, cv::Scalar(1) );	

	Mat mask_hor = Mat::zeros(bg.rows, bg.cols, CV_8U);
	Point pts_hor[7] = {
		Point(0, 0),
		Point(0, h * 2 + w),
		Point((h * 2 + w) / 2, (h * 2 + w) / 2),
		Point( h * 2 + w,  h * 2 + w),
		Point(h * 2 + w,0),
		Point((h * 2 + w) / 2, (h * 2 + w) / 2),
		Point(0, 0)

	};
	fillConvexPoly( mask_hor, pts_hor, 7, cv::Scalar(1) );*/	

	Mat mask_front = Mat::zeros(h, w, CV_8U);
	Point pts_front[] = {
		Point(0, 0),
		Point(w * 2 / 5 , h * 5 / 8),
		Point(w * 3 / 5 , h * 5 / 8),
		Point(w ,0)
	};
	fillConvexPoly( mask_front, pts_front, 4, cv::Scalar(1) );	
	
	Mat mask_rear = Mat::zeros(h, w, CV_8U);
	Point pts_rear[] = {
		Point(w / 11, h),
		Point(w * 2 / 5 , h * 3 / 8),
		Point(w * 3 / 5 , h * 3 / 8),
		Point(w * 10 / 11 ,h)
	};
	fillConvexPoly( mask_rear, pts_rear, 4, cv::Scalar(1) );

	Mat mask_left = Mat::zeros(w, h, CV_8U);
	Point pts_left[] = {
		Point(0, 0),
		Point(h * 5 / 8, w * 2 / 5),
		Point(h * 5 / 8, w * 3 / 5),
		Point(0, w)
	};
	fillConvexPoly( mask_left, pts_left, 4, cv::Scalar(1) );

	Mat mask_right = Mat::zeros(w, h, CV_8U);
	Point pts_right[] = {
		Point(h, 0),
		Point(h * 3 / 8, w * 2 / 5),
		Point(h * 3 / 8, w * 3 / 5),
		Point(h, w)
	};
	fillConvexPoly( mask_right, pts_right, 4, cv::Scalar(1) );	

	Rect whereRec_right(w + h - (move_together_ + move_right_), h, 720, 1280);
	destination_right.copyTo(bckg_right(whereRec_right), mask_right);
	
	Rect whereRec_left(0 +  move_together_ + move_left_, h, 720, 1280);
	destination_left.copyTo(bckg_left(whereRec_left),mask_left);

	Rect whereRec_front(h, 0 + move_together_ + move_front_, 1280, 720);
	destination_front.copyTo(bckg_front(whereRec_front),mask_front);

	Rect whereRec_rear(h, w + h - (move_together_ + move_rear_), 1280, 720);
	destination_rear.copyTo(bckg_rear(whereRec_rear), mask_rear);
	
	Mat vis_vert, vis_hor, vis_res, resized;
	bitwise_or(bckg_front, bckg_rear, vis_vert);
	bitwise_or(bckg_left, bckg_right, vis_hor);
	bitwise_or(vis_hor, vis_vert, vis_res);
	resize(vis_res,resized,Size(vis_res.cols/2,vis_res.rows/2),0, 0, INTER_NEAREST);
	auto end_merge  = chrono::high_resolution_clock::now();	
	

	auto start_imshow = chrono::high_resolution_clock::now();
	imshow("Result", resized);
	/*#pragma omp parallel num_threads(4)
	{
		#pragma omp sections
		{		
			#pragma omp section
			{
				imshow("Undistorted1", destination_front);
			}
			#pragma omp section
			{
				imshow("Left", destination_left);
			}
			#pragma omp section
			{
				imshow("Right", destination_right);
			}
			#pragma omp section
			{
				imshow("rear", destination_rear);
			}
		}
	}*/
	auto end_imshow = chrono::high_resolution_clock::now();
	
        if (waitKey(1) == 27) {
            break;
        }
	auto end_total  = chrono::high_resolution_clock::now();

	double cap_time = chrono::duration_cast<chrono::nanoseconds>(end_cap - start_cap).count();
	double undist_time = chrono::duration_cast<chrono::nanoseconds>(end_undist - start_undist).count();
	double perspect_time = chrono::duration_cast<chrono::nanoseconds>(end_perspect - start_perspect).count();
	double merge_time = chrono::duration_cast<chrono::nanoseconds>(end_merge - start_merge).count();
	double imshow_time = chrono::duration_cast<chrono::nanoseconds>(end_imshow - start_imshow).count();
	double total_time = chrono::duration_cast<chrono::nanoseconds>(end_total - start_total).count();
	cap_time *= 1e-9;
	undist_time *= 1e-9;
	perspect_time *= 1e-9;
	merge_time *= 1e-9;
	imshow_time *= 1e-9;
	total_time *= 1e-9;
	cout<<"Capture: "<<fixed<<cap_time<<setprecision(3)<<" sec  ";
	cout<<"Undistortion: "<<fixed<<undist_time<<setprecision(3)<<" sec  ";
	cout<<"Perspect: "<<fixed<<perspect_time<<setprecision(3)<<" sec  ";
	cout<<"Merge: "<<fixed<<merge_time<<setprecision(3)<<" sec  ";
	cout<<"Imshow: "<<fixed<<imshow_time<<setprecision(3)<<" sec  ";
	cout<<"Total: "<<fixed<<total_time<<setprecision(3)<<" sec  "<<endl;
    }
	}
    return 0;
}
