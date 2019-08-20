
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

int main()
{
	
	setUseOptimized(true);
	Mat cameraMatrix = (Mat_<double>(3,3) << 4.3526298745939096e+02, 0., 6.3950000000000000e+02, 0., 4.3415306567637083e+02, 3.5950000000000000e+02, 0., 0., 1.);
	Mat test = (Mat_<double>(3,3) << 4, 0, 8, 0, 4, 6, 0, 0, 1);
	Mat distCoeffs = (Mat_<double>(4,1) << -4.5133663311739985e-02, 2.1055083480493644e-02,
       -1.0118009701988350e-02, 0.);
	Mat frame1(Size(1280,720), CV_32FC1);
	Mat frame2(Size(1280,720), CV_32FC1);
	Mat frame3(Size(1280,720), CV_32FC1);
	Mat frame4(Size(1280,720), CV_32FC1);
	Mat view, undist1, undist2, undist3, undist4, newCamMat;
	Mat map1(Size(1280,720),CV_16SC2);
	Mat map2(Size(1280,720),CV_16UC1);
	Mat front = Mat::zeros(Size(1000,1500), CV_32FC1);
	Mat rear = Mat::zeros(Size(1000,1500), CV_32FC1);
	Mat left = Mat::zeros(Size(1000,1500), CV_32FC1);
	Mat right = Mat::zeros(Size(1000,1500), CV_32FC1);
	int h = 720;
	int w = 1280;
	Mat vis = Mat::zeros(Size((w+h*2),(w+h*2)),CV_32FC1);
	Mat vis_upper = Mat::zeros(Size((w+h*2),(w+h*2)),CV_32FC1);
	Mat vis_left = Mat::zeros(Size((w+h*2),(w+h*2)),CV_32FC1);
	Mat vis_right = Mat::zeros(Size((w+h*2),(w+h*2)),CV_32FC1);
	Mat vis_lower = Mat::zeros(Size((w+h*2),(w+h*2)),CV_32FC1);
 
	VideoCapture cap1("udpsrc port=50003 ! application/x-rtp,media=video,payload=26,clock-rate=90000,encoding-name=JPEG,framerate=20/1 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink ", CAP_GSTREAMER);
    VideoCapture cap2("udpsrc port=50004 ! application/x-rtp,media=video,payload=26,clock-rate=90000,encoding-name=JPEG,framerate=20/1 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink ", CAP_GSTREAMER);
	VideoCapture cap3("udpsrc port=50005 ! application/x-rtp,media=video,payload=26,clock-rate=90000,encoding-name=JPEG,framerate=20/1 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink ", CAP_GSTREAMER);
	VideoCapture cap4("udpsrc port=50006 ! application/x-rtp,media=video,payload=26,clock-rate=90000,encoding-name=JPEG,framerate=20/1 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink ", CAP_GSTREAMER);

	/*VideoCapture cap1("/home/aleksandr/birds_eye_view_system/test1.mp4");
	VideoCapture cap2("/home/aleksandr/birds_eye_view_system/test2.mp4");
	VideoCapture cap3("/home/aleksandr/birds_eye_view_system/test3.mp4");
	VideoCapture cap4("/home/aleksandr/birds_eye_view_system/test4.mp4");*/	

	if (!cap1.isOpened()) {
        cerr <<"VideoCapture 1 not opened"<<endl;
        exit(-1);
    }
	if (!cap2.isOpened()) {
        cerr <<"VideoCapture 2 not opened"<<endl;
        exit(-1);
    }
	if (!cap3.isOpened()) {
        cerr <<"VideoCapture 3 not opened"<<endl;
        exit(-1);
    }
	if (!cap4.isOpened()) {
        cerr <<"VideoCapture 4 not opened"<<endl;
        exit(-1);
    }
    	cap1.read(frame1);
	Size imageSize = frame1.size();
	fisheye::estimateNewCameraMatrixForUndistortRectify(cameraMatrix, distCoeffs, imageSize,Matx33d::eye(), newCamMat, 1);
        fisheye::initUndistortRectifyMap(cameraMatrix, distCoeffs, Matx33d::eye(), newCamMat, imageSize,CV_16SC2, map1, map2);
	
	Mat destination1(frame1.size(), CV_32FC1);
	Mat destination2(frame1.size(), CV_32FC1);
	Mat destination3(frame1.size(), CV_32FC1);
	Mat destination4(frame1.size(), CV_32FC1);
	int alpha_upper_ = 97, beta_upper_ = 90, gamma_upper_ = 90, f_upper_ = 279, d_upper_ = 500;
	int alpha_left_ = 106, beta_left_ = 90, gamma_left_ = 90, f_left_ = 279, d_left_ = 500;
	int alpha_right_ = 105, beta_right_ = 90, gamma_right_ = 90, f_right_ = 279, d_right_ = 500;
	int alpha_lower_ = 106, beta_lower_ = 90, gamma_lower_ = 90, f_lower_ = 279, d_lower_ = 500;
	int move_upper_ = 171, move_left_ = 162, move_right_ = 103, move_lower_ = 165, move_together_ = 645;

	namedWindow("Result", 1);
	namedWindow("tune_upper");
	namedWindow("tune_left");
	namedWindow("tune_right");
	namedWindow("tune_lower");
	namedWindow("move");

	createTrackbar("Alpha","tune_upper",&alpha_upper_,180);
	createTrackbar("Beta","tune_upper",&beta_upper_,180);
	createTrackbar("Gamma","tune_upper",&gamma_upper_,180);
	createTrackbar("f","tune_upper",&f_upper_,2000);
	createTrackbar("Distance","tune_upper",&d_upper_,2000);

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

	createTrackbar("Alpha","tune_lower",&alpha_lower_,180);
	createTrackbar("Beta","tune_lower",&beta_lower_,180);
	createTrackbar("Gamma","tune_lower",&gamma_lower_,180);
	createTrackbar("f","tune_lower",&f_lower_,2000);
	createTrackbar("Distance","tune_lower",&d_lower_,2000);

	createTrackbar("Move upper","move",&move_upper_,400);
	createTrackbar("Move left","move",&move_left_,400);
	createTrackbar("Move right","move",&move_right_,400);
	createTrackbar("Move lower","move",&move_lower_,400);
	createTrackbar("Move together","move",&move_together_,700);
	
	double focalLength, dist, alpha, beta, gamma;
	cout<<"Optimization"<<useOptimized()<<'\n';
	int counter = 0;
    while (true) {
	
	auto start_total = chrono::high_resolution_clock::now();
	auto start_cap  = chrono::high_resolution_clock::now();
	#pragma omp parallel num_threads(4)
	{
		#pragma omp sections
		{		
			#pragma omp section
			{
				cap1.read(frame1);
			}
			#pragma omp section
			{
				cap2.read(frame2);
			}
			#pragma omp section
			{
				cap3.read(frame3);
			}
			#pragma omp section
			{
				cap4.read(frame4);
			}
		}
	}

	counter++;
	if(counter%2 == 1){

	auto end_cap  = chrono::high_resolution_clock::now();
	
	auto start_undist  = chrono::high_resolution_clock::now();
	
		remap(frame1, undist1, map1, map2, INTER_CUBIC, BORDER_CONSTANT);
		remap(frame2, undist2, map1, map2, INTER_CUBIC, BORDER_CONSTANT);	
		remap(frame3, undist3, map1, map2, INTER_CUBIC, BORDER_CONSTANT);
		remap(frame4, undist4, map1, map2, INTER_CUBIC, BORDER_CONSTANT);
			
	auto end_undist  = chrono::high_resolution_clock::now();

	auto start_perspect  = chrono::high_resolution_clock::now();
		
		alpha_upper_ = getTrackbarPos("Alpha","tune_upper");
		beta_upper_ = getTrackbarPos("Beta","tune_upper");
		gamma_upper_ = getTrackbarPos("Gamma","tune_upper");
		f_upper_ = getTrackbarPos("f","tune_upper");
		d_upper_ = getTrackbarPos("Distance","tune_upper");

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

		alpha_lower_ = getTrackbarPos("Alpha","tune_lower");
		beta_lower_ = getTrackbarPos("Beta","tune_lower");
		gamma_lower_ = getTrackbarPos("Gamma","tune_lower");
		f_lower_ = getTrackbarPos("f","tune_lower");
		d_lower_ = getTrackbarPos("Distance","tune_lower");

		move_upper_ = getTrackbarPos("Move upper","move");
		move_left_ = getTrackbarPos("Move left","move");
		move_right_ = getTrackbarPos("Move right","move");
		move_lower_ = getTrackbarPos("Move lower","move");
		move_together_ = getTrackbarPos("Move together","move");

	#pragma omp parallel num_threads(4)
	{
		#pragma omp sections
		{		
			#pragma omp section
			{
				Mat transform_upper = getTransformation(1280, 720, alpha_upper_, beta_upper_, gamma_upper_, f_upper_, d_upper_);
				warpPerspective(undist3, destination1, transform_upper, imageSize, INTER_NEAREST );
			}
			#pragma omp section
			{
				Mat transform_left = getTransformation(1280, 720, alpha_left_, beta_left_, gamma_left_, f_left_, d_left_);
				warpPerspective(undist4, destination2, transform_left, imageSize, INTER_NEAREST );
			}
			#pragma omp section
			{
				Mat transform_right = getTransformation(1280, 720, alpha_right_, beta_right_, gamma_right_, f_right_, d_right_);
				warpPerspective(undist1, destination3, transform_right, imageSize, INTER_NEAREST );
			}
			#pragma omp section
			{
				Mat transform_lower = getTransformation(1280, 720, alpha_lower_, beta_lower_, gamma_lower_, f_lower_, d_lower_);
				warpPerspective(undist2, destination4, transform_lower, imageSize, INTER_NEAREST );
			}
		}
	}

	auto end_perspect  = chrono::high_resolution_clock::now();	
	
	
	Rect roi_upper = Rect(300,300,destination1.cols+300,destination1.rows+300);
	Mat roiImg(vis_upper,roi_upper);
	destination1.copyTo(roiImg);
	imshow("ROI_upper", roiImg);
	cout<<roiImg.size();
	

	//Rect roi_lower = Rect(720,2000,2000,2720);
	//Mat roiImg_lower;
	//roiImg_lower = vis_lower(roi_lower);
	//destination4.copyTo(roiImg_lower);
	//imshow("ROI_lower", roiImg_lower);
	//cout<<roi_lower.size();

	auto start_imshow = chrono::high_resolution_clock::now();

	#pragma omp parallel num_threads(4)
	{
		#pragma omp sections
		{		
			#pragma omp section
			{
				imshow("Undistorted1", destination1);
			}
			#pragma omp section
			{
				imshow("Undistorted2", destination2);
			}
			#pragma omp section
			{
				imshow("Undistorted3", destination3);
			}
			#pragma omp section
			{
				imshow("Undistorted4", destination4);
			}
		}
	}
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
