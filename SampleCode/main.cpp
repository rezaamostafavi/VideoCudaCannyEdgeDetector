
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

//int main(int argc, char *argv[])
//{
//	Mat img = imread("f:/1.jpg");
//	Mat gray;
//	cvtColor(img, gray, CV_BGR2GRAY);
//	wrapper(gray.data);
//	imshow("GrayScaled", img);
//	waitKey(0);
//}