
#include <opencv2/opencv.hpp>
#include <iostream>
#include <bits/stdc++.h>
#include <chrono>

#define PI 3.1415926

using namespace std;
using namespace cv;

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

int main()
{	
	setUseOptimized(true);
	Mat cameraMatrix = (Mat_<double>(3,3) << 4.3526298745939096e+02, 0., 6.3950000000000000e+02, 0., 4.3415306567637083e+02, 3.5950000000000000e+02, 0., 0., 1.);
	Mat test = (Mat_<double>(3,3) << 4, 0, 8, 0, 4, 6, 0, 0, 1);
	Mat distCoeffs = (Mat_<double>(4,1) << -4.5133663311739985e-02, 2.1055083480493644e-02,
       -1.0118009701988350e-02, 0.);
	Mat frame_left(Size(1280,720), CV_32FC3);
	Mat frame_right(Size(1280,720), CV_32FC3);
	Mat frame_front(Size(1280,720), CV_32FC3);
	Mat frame_rear(Size(1280,720), CV_32FC3);
	Mat view, newCamMat;
	Mat undist_left(Size(1280,720),CV_32FC3);
	Mat undist_right(Size(1280,720),CV_32FC3);
	Mat undist_front(Size(1280,720),CV_32FC3);
	Mat undist_rear(Size(1280,720),CV_32FC3);
	Mat map1(Size(1280,720),CV_16SC2);
	Mat map2(Size(1280,720),CV_16UC1);
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
 
	VideoCapture cap_right("udpsrc port=50003 ! application/x-rtp,media=video,payload=26,clock-rate=90000,encoding-name=JPEG,framerate=20/1 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink ", CAP_GSTREAMER);
    VideoCapture cap_rear("udpsrc port=50004 ! application/x-rtp,media=video,payload=26,clock-rate=90000,encoding-name=JPEG,framerate=20/1 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink ", CAP_GSTREAMER);
	VideoCapture cap_front("udpsrc port=50005 ! application/x-rtp,media=video,payload=26,clock-rate=90000,encoding-name=JPEG,framerate=20/1 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink ", CAP_GSTREAMER);
	VideoCapture cap_left("udpsrc port=50006 ! application/x-rtp,media=video,payload=26,clock-rate=90000,encoding-name=JPEG,framerate=20/1 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink ", CAP_GSTREAMER);

	/*VideoCapture cap1("/home/aleksandr/birds_eye_view_system/test1.mp4");
	VideoCapture cap_rear("/home/aleksandr/birds_eye_view_system/test2.mp4");
	VideoCapture cap_front("/home/aleksandr/birds_eye_view_system/test3.mp4");
	VideoCapture cap_left("/home/aleksandr/birds_eye_view_system/test4.mp4");*/	

	if (!cap_right.isOpened()) {
        cerr <<"VideoCapture right not opened"<<endl;
        exit(-1);
    }
	if (!cap_rear.isOpened()) {
        cerr <<"VideoCapture 2 not opened"<<endl;
        exit(-1);
    }
	if (!cap_front.isOpened()) {
        cerr <<"VideoCapture 3 not opened"<<endl;
        exit(-1);
    }
	if (!cap_left.isOpened()) {
        cerr <<"VideoCapture 4 not opened"<<endl;
        exit(-1);
    }
    	cap_right.read(frame_right);
	Size imageSize = frame_right.size();
	fisheye::estimateNewCameraMatrixForUndistortRectify(cameraMatrix, distCoeffs, imageSize,Matx33d::eye(), newCamMat, 1);
        fisheye::initUndistortRectifyMap(cameraMatrix, distCoeffs, Matx33d::eye(), newCamMat, imageSize,CV_16SC2, map1, map2);
	
	Mat destination_right(frame_right.size(), CV_32FC3);
	Mat destination_left(frame_left.size(), CV_32FC3);
	Mat destination_front(frame_front.size(), CV_32FC3);
	Mat destination_rear(frame_rear.size(), CV_32FC3);
	int alpha_front_ = 129, beta_front_ = 90, gamma_front_ = 90, f_front_ = 300, d_front_ = 500;
	int alpha_left_ = 129, beta_left_ = 90, gamma_left_ = 90, f_left_ = 300, d_left_ = 500;
	int alpha_right_ = 129, beta_right_ = 90, gamma_right_ = 90, f_right_ = 300, d_right_ = 500;
	int alpha_rear_ = 129, beta_rear_ = 90, gamma_rear_ = 90, f_rear_ = 300, d_rear_ = 500;
	int move_front_ = 0, move_left_ = 0, move_right_ = 0, move_rear_ = 0, move_together_ = 645;

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
	cout<<"Optimization"<<useOptimized()<<'\n';
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
	
		remap(frame_right, undist_right, map1, map2, INTER_CUBIC, BORDER_CONSTANT);
		remap(frame_rear, undist_rear, map1, map2, INTER_CUBIC, BORDER_CONSTANT);	
		remap(frame_front, undist_front, map1, map2, INTER_CUBIC, BORDER_CONSTANT);
		remap(frame_left, undist_left, map1, map2, INTER_CUBIC, BORDER_CONSTANT);
			
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

	#pragma omp parallel num_threads(4)
	{
		#pragma omp sections
		{		
			#pragma omp section
			{
				Mat transform_front = getTransformation(1280, 720, alpha_front_, beta_front_, gamma_front_, f_front_, d_front_);
				warpPerspective(undist_front, destination_front, transform_front, imageSize, INTER_NEAREST );
			}
			#pragma omp section
			{
				Mat transform_left = getTransformation(1280, 720, alpha_left_, beta_left_, gamma_left_, f_left_, d_left_);
				warpPerspective(undist_left, destination_left, transform_left, imageSize, INTER_NEAREST );
			}
			#pragma omp section
			{
				Mat transform_right = getTransformation(1280, 720, alpha_right_, beta_right_, gamma_right_, f_right_, d_right_);
				warpPerspective(undist_right, destination_right, transform_right, imageSize, INTER_NEAREST );
			}
			#pragma omp section
			{
				Mat transform_rear = getTransformation(1280, 720, alpha_rear_, beta_rear_, gamma_rear_, f_rear_, d_rear_);
				warpPerspective(undist_rear, destination_rear, transform_rear, imageSize, INTER_NEAREST );
			}
		}
	}

	auto end_perspect  = chrono::high_resolution_clock::now();	

	destination_left = rotateBound(destination_left, 270);
	destination_right = rotateBound(destination_right, 90);
	flip(destination_left, destination_left, 1);
	flip(destination_right, destination_right, 1);
	flip(destination_rear, destination_rear, 0);
	flip(destination_rear, destination_rear, 1);	

	Mat bckg_left = bg;
	Mat bckg_right = bg;
	Mat bckg_front = bg;
	Mat bckg_rear = bg;
	
	Mat mask_vert = Mat::zeros(bg.rows, bg.cols, CV_8U);
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
	fillConvexPoly( mask_hor, pts_hor, 7, cv::Scalar(1) );	

	Mat mask_front = Mat::zeros(h, w, CV_8U);
	Point pts_front[] = {
		Point(0, 0),
		Point(w / 2, h),
		Point(w ,0)
	};
	fillConvexPoly( mask_front, pts_front, 3, cv::Scalar(1) );	
	
	Mat mask_rear = Mat::zeros(h, w, CV_8U);
	Point pts_rear[] = {
		Point(0, h),
		Point(w / 2, 0),
		Point(w ,h)
	};
	fillConvexPoly( mask_rear, pts_rear, 3, cv::Scalar(1) );	
	
	Rect whereRec_right(w + h - (move_together_ + move_right_), h, 720, 1280);
	destination_right.copyTo(bckg_right(whereRec_right));
	
	Rect whereRec_left(0 +  move_together_ + move_left_, h, 720, 1280);
	destination_left.copyTo(bckg_left(whereRec_left));

	Rect whereRec_front(h, 0 + move_together_ + move_front_, 1280, 720);
	destination_front.copyTo(bckg_front(whereRec_front),mask_front);

	Rect whereRec_rear(h, w + h - (move_together_ + move_rear_), 1280, 720);
	destination_rear.copyTo(bckg_rear(whereRec_rear), mask_rear);
	
	Mat vis_vert, vis_hor, vis_res, resized;
	bitwise_or(bckg_front, bckg_rear, vis_vert);
	bitwise_or(bckg_left, bckg_right, vis_hor);
	bitwise_or(vis_vert, vis_hor, vis_res);
	resize(vis_res,resized,Size(vis_res.cols/2,vis_res.rows/2),0, 0, INTER_NEAREST);	
	

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
	double imshow_time = chrono::duration_cast<chrono::nanoseconds>(end_imshow - start_imshow).count();
	double total_time = chrono::duration_cast<chrono::nanoseconds>(end_total - start_total).count();
	cap_time *= 1e-9;
	undist_time *= 1e-9;
	perspect_time *= 1e-9;
	imshow_time *= 1e-9;
	total_time *= 1e-9;
	cout<<"Capture: "<<fixed<<cap_time<<setprecision(9)<<" sec  ";
	cout<<"Undistortion: "<<fixed<<undist_time<<setprecision(9)<<" sec  ";
	cout<<"Perspect: "<<fixed<<perspect_time<<setprecision(9)<<" sec  ";
	cout<<"Imshow: "<<fixed<<imshow_time<<setprecision(9)<<" sec  ";
	cout<<"Total: "<<fixed<<total_time<<setprecision(9)<<" sec  "<<endl;
	
    }
	}

    return 0;
}
