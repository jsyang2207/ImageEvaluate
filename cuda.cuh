#pragma once
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudev/ptr2d/gpumat.hpp>
#include <opencv2/core/core.hpp>
#include "cuda.h"
#include <iostream>
#include <cufft.h>
#include "cublas_v2.h"
#ifdef __cplusplus 
using namespace cv;
extern "C" {//<-- extern ½ÃÀÛ

#endif
	class CGPUACC
	{
	public:

		CGPUACC(void);

		virtual ~CGPUACC(void);
		int sum_cuda(int a, int b, int* c);
		double getSigma(Mat& m, int i, int j, int block_size, double* sd);
		double getSSIM(Mat& img_src, Mat& img_compressed, int block_size, double* ssim);
		double getCov(Mat& m1, Mat& m2, int i, int j, int block_size, double* sd_ro);
		int frame_cuda(Mat& img_src);
	};
#ifdef __cplusplus 
}


#endif

