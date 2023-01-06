#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Inversion_CUDA.h"

#include <algorithm>
#include <chrono>
#include <vector>
using namespace std::chrono;

using namespace std;
using namespace cv;


int main() {
	Mat Input_Image = imread("Test_Image.png");

	cout << "Height: " << Input_Image.cols << ", Width: " << Input_Image.rows << ", Channels: " << Input_Image.channels() << endl;

	auto start = high_resolution_clock::now();

	Image_Inversion_CUDA(Input_Image.data, Input_Image.cols, Input_Image.rows, Input_Image.channels());

	auto stop = high_resolution_clock::now();
	
	auto duration = duration_cast<microseconds>(stop - start);

	double timee = duration.count();

	cout << "Time taken by function: "
		<< timee/1000000 << " Seconds" << endl;

	imwrite("Inverted_Image.png", Input_Image);
	system("pause");
	return 0;
}