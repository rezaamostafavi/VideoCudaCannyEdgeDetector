#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cmath>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace std;
using namespace cv;

#define M_PI       3.14159265358979323846   // pi

 __device__ void useFilterCuda(uchar *img_in, uchar *img_out, double* filterIn, int sizeFilter, int rows, int cols) {
	int index = threadIdx.x;
	int stride = blockDim.x;
	//printf("index , stride = %d,%d\n", index, stride);
	//int max = index*stride > rows*cols ? index*stride : rows*cols;
	for (int i = index; i < rows*cols; i += stride)
	{
		int x =(int)(i / cols);
		int y = i % cols;
		if (x + sizeFilter < rows && y + sizeFilter < cols) {
			double sum = 0;
			for (int xf = 0; xf < sizeFilter; xf++)
				for (int yf = 0; yf < sizeFilter; yf++)
				{
					int index = ((x + xf)*cols) + (y + yf);
					int fIndex = xf*sizeFilter + yf;
					sum += img_in[index] * filterIn[fIndex];
				}
			img_out[i] = sum;
		}
		//printf("%d,", i);
	}
}

__device__ void sobelCuda(uchar *imgIn, uchar *imgOut, float *imgAngles, int rows, int cols) {
	//Sobel X Filter
	double xFilter[] = { -1.0f, 0, 1.0f, -2.0f, 0, 2.0f, -1.0f, 0, 1.0f };

	//Sobel Y Filter
	double yFilter[] = { 1.0f, 2.0f, 1.0f, 0, 0, 0, -1.0f, -2.0f, -1.0f };

	int index = threadIdx.x;
	int stride = blockDim.x;
	//int max = index*stride > rows*cols ? index*stride : rows*cols;
	for (int i = index; i < rows*cols; i += stride)
	{
		double sumx = 0;
		double sumy = 0;
		int xIndex = i / cols;
		int yIndex = i % cols;
		if (xIndex + 3 < rows && yIndex + 3 < cols) {
			for (int x = 0; x < 3; x++)
				for (int y = 0; y < 3; y++)
				{
					int fIndex = x * 3 + y;
					int index = (xIndex + x)*cols + yIndex + y;
					sumx += xFilter[fIndex] * (imgIn[index]); //Sobel_X Filter Value
					sumy += yFilter[fIndex] * (imgIn[index]); //Sobel_Y Filter Value
				}
			double sumxsq = sumx*sumx;
			double sumysq = sumy*sumy;

			double sq2 = sqrt(sumxsq + sumysq);

			if (sq2 > 255) //Unsigned Char Fix
				sq2 = 255;
			imgOut[i] = sq2;

			if (sumx == 0) //Arctan Fix
				imgAngles[i] = 90.0f;
			else
				imgAngles[i] = atan(sumy / sumx);
		}
		else
			imgAngles[i] = 90.0f;
	}
}

__device__ void nonMaxSupp(uchar* img_in, uchar* img_out, float *angles, int rows, int cols)
{
	int index = threadIdx.x;
	int stride = blockDim.x;
	int max = index*stride > rows*cols - cols - rows ? index*stride : rows*cols - cols - rows;
	int start = index > cols ? index : cols;
	int mSize = rows*cols;
	for (int i = index; i < mSize; i += stride) {
		float targentData = angles[i];
		int x = i / cols;
		int y = i % cols;
		int index = (x - 1)*cols + y - 1;
		if (index > 0 && index < mSize) {
			img_out[index] = img_in[i];
			//Horizontal Edge
			if ((targentData > -22.5 && targentData <= 22.5) || (targentData > 157.5 && targentData <= -157.5)) {
				if (img_in[i] < img_in[i + 1] || img_in[i] < img_in[i - 1])
					img_out[index] = 0;
			}
			//Vertical Edge
			if (((-112.5 < targentData) && (targentData <= -67.5)) || ((67.5 < targentData) && (targentData <= 112.5)))
			{
				if (x + 1 < rows && x - 1 > 0 && (img_in[i] < img_in[(x + 1)*cols + (y)] || img_in[i] < img_in[(x - 1)*cols + y]))
					img_out[index] = 0;
			}

			//-45 Degree Edge
			if (((-67.5 < targentData) && (targentData <= -22.5)) || ((112.5 < targentData) && (targentData <= 157.5)))
			{
				if (y + 1 < cols && x - 1 > 0 && x + 1 < rows && y - 1 > 0 &&
					(img_in[i] < img_in[(x - 1)*cols + (y + 1)] || img_in[i] < img_in[(x + 1)*cols + (y - 1)]))
					img_out[index] = 0;
			}

			//45 Degree Edge
			if (((-157.5 < targentData) && (targentData <= -112.5)) || ((22.5 < targentData) && (targentData <= 67.5)))
			{
				if (y + 1 < cols && x - 1 > 0 && x + 1 < rows && y - 1 > 0 &&
					(img_in[i] < img_in[(x + 1)*cols + (y + 1)] || img_in[i] < img_in[(x - 1)*cols + (y - 1)]))
					img_out[index] = 0;
			}
		}
	}
}

__device__ void threshold(uchar *img_in, uchar *img_out, int low, int high, int rows, int cols)
{
	if (low > 255)
		low = 255;
	if (high > 255)
		high = 255;

	int index = threadIdx.x;
	int stride = blockDim.x;
	int max = index*stride > rows*cols ? index*stride : rows*cols;
	for (int i = index; i < rows*cols; i += stride)
	{
		img_out[i] = img_in[i];
		if (img_out[i] > high)
			img_out[i] = 255;
		else if (img_out[i] < low)
			img_out[i] = 0;
		else
		{
			bool anyHigh = false;
			bool anyBetween = false;
			for (int x = 0; x < 3 * 3; x++)
			{
				int index = (i + ((x / 3) - 1)*cols) + ((i%cols) + (x % 3) - 1);
				if (index < 0 || index >= cols*rows) //Out of bounds
					continue;
				else
				{
					if (img_out[index] > high)
					{
						img_out[i] = 255;
						anyHigh = true;
						break;
					}
					else if (img_out[index] <= high && img_out[index] >= low)
						anyBetween = true;
				}
				if (anyHigh)
					break;
			}
			if (!anyHigh && anyBetween)
				for (int x = 0; x < 5 * 5; x++)
				{
					int index = (i + ((x / 5) - 2)*cols) + ((i%cols) + (x % 5) - 2);
					if (index < 0 || index >= cols*rows) //Out of bounds
						continue;
					else
					{
						if (img_out[index] > high)
						{
							img_out[i] = 255;
							anyHigh = true;
							break;
						}
					}

					if (anyHigh)
						break;
				}
			if (!anyHigh)
				img_out[i] = 0;
		}
	}
}

__global__ void canny(uchar *imgIn, uchar *imgGaussian, uchar *imgSober, uchar *imgNonMax, uchar *imgFinal,
	float *angles, double *filterIn, int sizeFilter, int rows, int cols) {
	int index = threadIdx.x;
	int stride = blockDim.x;
	useFilterCuda(imgIn, imgGaussian, filterIn, sizeFilter, rows, cols);//Gaussian filter
	sobelCuda(imgGaussian, imgSober, angles, rows, cols);//Finding the intensity (Sobel) 
	nonMaxSupp(imgSober, imgNonMax, angles, rows, cols);//Non-maximum suppression
	threshold(imgNonMax, imgFinal, 20, 40, rows, cols);//Double threshold
}

double* createFilter(int row, int column, double sigmaIn)
{
	vector<vector<double>> filter;

	for (int i = 0; i < row; i++)
	{
		vector<double> col;
		for (int j = 0; j < column; j++)
		{
			col.push_back(-1);
		}
		filter.push_back(col);
	}

	float coordSum = 0;
	float constant = 2.0 * sigmaIn * sigmaIn;

	// Sum is for normalization
	float sum = 0.0;

	for (int x = -row / 2; x <= row / 2; x++)
	{
		for (int y = -column / 2; y <= column / 2; y++)
		{
			coordSum = (x*x + y*y);
			filter[x + row / 2][y + column / 2] = (exp(-(coordSum) / constant)) / (M_PI * constant);
			sum += filter[x + row / 2][y + column / 2];
		}
	}

	// Normalize the Filter
	for (int i = 0; i < row; i++)
		for (int j = 0; j < column; j++)
			filter[i][j] /= sum;
	double *mFilter = new double[row*column];
	for (int i = 0; i<filter.size(); i++)
	{
		for (int j = 0; j<filter[i].size(); j++)
		{
			int index = i*row + j;
			mFilter[index] = filter[i][j];
		}
	}
	return mFilter;

}

int main() {
	uchar *img_sober, *img_non, *img_final, *img_filter,*img_in;
	uchar *cSoberImg, *cNonImg, *cFinalImg, *cFilterImg, *cInImg;
	float *angles,*cAngles;
	double *filterIn, *cFilter;
	int sizeFilter = 5;

	string ip = "http://127.0.0.1:8081/video.mjpg";
	VideoCapture capture(ip);
	Mat img;

	namedWindow("Original", WINDOW_NORMAL);
	resizeWindow("Original", 800, 600);
	namedWindow("Canny", WINDOW_NORMAL);
	resizeWindow("Canny", 800, 600);

	while (true)
	{

		capture.read(img);
		//Mat img = imread("f:/1.jpg");
		Mat gray;
		cvtColor(img, gray, CV_BGR2GRAY);

		Mat filterImage = Mat(img.rows, img.cols, CV_8UC1);
		Mat sobelImage = Mat(img.rows, img.cols, CV_8UC1);
		Mat nonMaxImage = Mat(img.rows, img.cols, CV_8UC1);
		Mat finalImage = Mat(img.rows, img.cols, CV_8UC1);

		long size = sizeof(uchar) * img.rows * img.cols;
		int cSizeFilter = sizeof(double)*sizeFilter*sizeFilter;
		int sizeAngles = sizeof(float) * img.rows * img.cols;
		angles = new float[img.rows*img.cols];

		filterIn = createFilter(sizeFilter, sizeFilter, 1);

		// Allocate host memory
		img_sober = (uchar*)malloc(size);
		img_filter = (uchar*)malloc(size);
		img_final = (uchar*)malloc(size);
		img_non = (uchar*)malloc(size);

		// Allocate device memory 
		cudaMalloc((void**)&cInImg, size);
		cudaMalloc((void**)&cSoberImg, size);
		cudaMalloc((void**)&cNonImg, size);
		cudaMalloc((void**)&cFilterImg, size);
		cudaMalloc((void**)&cFinalImg, size);
		cudaMalloc((void**)&cAngles, sizeAngles);
		cudaMalloc((void**)&cFilter, cSizeFilter);

		// Transfer data from host to device memory
		cudaMemcpy(cInImg, gray.data, size, cudaMemcpyHostToDevice);
		cudaMemcpy(cFilterImg, filterImage.data, size, cudaMemcpyHostToDevice);
		cudaMemcpy(cSoberImg, sobelImage.data, size, cudaMemcpyHostToDevice);
		cudaMemcpy(cNonImg, nonMaxImage.data, size, cudaMemcpyHostToDevice);
		cudaMemcpy(cFinalImg, finalImage.data, size, cudaMemcpyHostToDevice);
		cudaMemcpy(cFilter, filterIn, cSizeFilter, cudaMemcpyHostToDevice);

		// Executing kernel 
		canny << <1, 256 >> > (cInImg, cFilterImg, cSoberImg, cNonImg, cFinalImg, cAngles, cFilter, sizeFilter, img.rows, img.cols);

		// Transfer data back to host memory
		cudaMemcpy(img_filter, cFilterImg, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(img_sober, cSoberImg, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(img_non, cNonImg, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(img_final, cFinalImg, size, cudaMemcpyDeviceToHost);

		filterImage.data = img_filter;
		sobelImage.data = img_sober;
		nonMaxImage.data = img_non;
		finalImage.data = img_final;

		// Deallocate device memory
		cudaFree(cInImg);
		cudaFree(cFilterImg);
		cudaFree(cSoberImg);
		cudaFree(cNonImg);
		cudaFree(cFinalImg);
		cudaFree(cFilter);
		cudaFree(cAngles);

		//show Images
		
		imshow("Original", img);
		imshow("Canny", finalImage);
		/*namedWindow("GrayScaled", WINDOW_NORMAL);
		namedWindow("Filter", WINDOW_NORMAL);
		namedWindow("Sober", WINDOW_NORMAL);
		namedWindow("NonMax", WINDOW_NORMAL);

		resizeWindow("GrayScaled", 800, 600);
		resizeWindow("Filter", 800, 600);
		resizeWindow("Sober", 800, 600);
		resizeWindow("NonMax", 800, 600);

		imshow("GrayScaled", gray);
		imshow("Filter", filterImage);
		imshow("Sober", sobelImage);
		imshow("NonMax", nonMaxImage);*/

		int c = waitKey(10);
		if ((char)c == 'c') { break; }
	}
	// Deallocate host memory
	free(img_sober);
	free(img_filter);
	free(img_final);
	free(img_non);
}
