#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
#define C1 (float) (0.01 * 255 * 0.01  * 255)
#define C2 (float) (0.03 * 255 * 0.03  * 255)
double getssim(Mat& img_src, Mat& img_compressed, int block_size);
double sigma(Mat& m, int i, int j, int block_size);
double cov(Mat& m1, Mat& m2, int i, int j, int block_size);