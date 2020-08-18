#include "cuda.cuh"

#include <stdio.h>
#include <stdlib.h>

double C1 = (0.01 * 255) * (0.01 * 255);
double C2 = (0.03 * 255) * (0.03 * 255);
CGPUACC::CGPUACC(void)
{
}
CGPUACC::~CGPUACC(void)
{
}

__global__ void sum_kernel(int a, int b, int* c) {
	*c = a + b;
}
__global__ void sigma(unsigned char* input, int i, int j, int block_size, double* sd)
{
	//const int xindex = blockidx.x * blockdim.x + threadidx.x;
	//const int yindex = blockidx.y * blockdim.y + threadidx.y;
	unsigned char* img;
	int sumimg = 30;
	int suminput = 30*30;
	
	/*cudaMalloc<unsigned char>(&img, (block_size) * (block_size));
	for (int k = 0; k < block_size * block_size; k++) {
		img[k] = input[i * j + k] * input[i * j + k];
	}
	
	for (int k = 0; k < block_size * block_size; k++) {
		sumimg += img[k];
		suminput += input[i * j + k];
	}*/

	double avg = sumimg / (block_size * block_size);
	double avg_2 = suminput / (block_size * block_size);
	
	//*sd = 10.0;
	*sd = sqrt(avg_2 - avg);
	//*sd = i + j;
	//char a =input[i*j];

	//Mat m;
	//Mat m_tmp = m(Range(i, i + block_size), Range(j, j + block_size));
	//Mat m_squared(block_size, block_size, CV_64F);
	//multiply(m_tmp, m_tmp, m_squared);
	//double avg = mean(m_tmp)[0];
	//double avg_2 = mean(m_squared)[0];
	//*sd = sqrt(avg_2 - avg * avg);

}



int CGPUACC::sum_cuda(int a, int b, int* c) {
	int* f;
	cudaMalloc((void**)&f, sizeof(int) * 1);
	cudaMemcpy(f, c, sizeof(int) * 1, cudaMemcpyHostToDevice);
	sum_kernel << <1, 1 >> > (a, b, f);
	cudaMemcpy(c, f, sizeof(int) * 1, cudaMemcpyDeviceToHost);

	cudaFree(f);
	return true;
}


int CGPUACC::frame_cuda(Mat& img_src) {
	int* f;
	float* A;
	float* dA;
	uchar* gpu_src, * gpu_dst;

	CUdeviceptr cuDevicptr;
	size_t pitch;
	int width = 1920;
	int height = 1080;
	cudaMallocPitch(&gpu_src,&pitch, width *sizeof(uchar)*3, height);
	cudaMemcpy2D(gpu_src, pitch, img_src.ptr<uchar>(), width * sizeof(uchar), width * sizeof(uchar)*3, height, cudaMemcpyHostToDevice);

	//cudaMallocPitch(&dA,&pitch,1920*3,1080);

	//cudaMemcpy2D(dA,pitch,A,sizeof(float)*1,1920*3,1080,cudaMemcpyHostToDevice);
	
	


	/*cudaMalloc((void**)&f, sizeof(int) * 1);
	cudaMemcpy(f, c, sizeof(int) * 1, cudaMemcpyHostToDevice);
	sum_kernel << <1, 1 >> > (a, b, f);
	cudaMemcpy(c, f, sizeof(int) * 1, cudaMemcpyDeviceToHost);*/

	cudaFree(f);
	return true;
}


//__global__ void cov(Mat& m1, Mat& m2, int i, int j, int block_size, double* sd_ro)
//{
//	Mat m3 = Mat::zeros(block_size, block_size, m1.depth());
//	Mat m1_tmp = m1(Range(i, i + block_size), Range(j, j + block_size));
//	Mat m2_tmp = m2(Range(i, i + block_size), Range(j, j + block_size));
//	
//	multiply(m1_tmp, m2_tmp, m3);
//
//	double avg_ro = mean(m3)[0];
//	double avg_r = mean(m1_tmp)[0];
//	double avg_o = mean(m2_tmp)[0];
//
//	*sd_ro = avg_ro - avg_o * avg_r;
//}

//__global__ void f_ssim(Mat& img_src, Mat& img_compressed, int block_size, double* ssim)
//{
//
//	int nbBlockPerHeight = img_src.rows / block_size;
//	int nbBlockPerWidth = img_src.cols / block_size;
//	CGPUACC cgpuacc;
//	for (int k = 0; k < nbBlockPerHeight; k++)
//	{
//		for (int l = 0; l < nbBlockPerWidth; l++)
//		{
//			int m = k * block_size;
//			int n = l * block_size;
//			double* osd;
//			double* rsd;
//			double* sd_ro;
//			double avg_o = mean(img_src(Range(k, k + block_size), Range(l, l + block_size)))[0];
//			double avg_r = mean(img_compressed(Range(k, k + block_size), Range(l, l + block_size)))[0];
//			
//			//sigma(img_src, m, n, block_size, osd);
//			//sigma(img_compressed, m, n, block_size, rsd);
//
//			double sigma_o = *osd;
//			double sigma_r = *rsd;
//			
//			//cov(img_src, img_compressed, m, n, block_size, sd_ro);
//			//cov << <1, 1 >> > (img_src, img_compressed, m, n, block_size, sd_ro);
//			double sigma_ro = *sd_ro;
//
//			*ssim += ((2 * avg_o * avg_r + C1) * (2 * sigma_ro + C2)) / ((avg_o * avg_o + avg_r * avg_r + C1) * (sigma_o * sigma_o + sigma_r * sigma_r + C2));
//
//		}
//	}
//
//	*ssim /= nbBlockPerHeight * nbBlockPerWidth;
//}

double CGPUACC::getSigma(Mat& m, int i, int j, int block_size, double* sd) {
	
	unsigned char* d_img_src;
	//double* d_img_compressed;
	cudaMalloc<unsigned char>(&d_img_src, (m.rows) * (m.cols) );
	//cudaMalloc((void**)&d_img_compressed, (img_compressed.rows) * (img_compressed.cols) * sizeof(double));
	cudaMemcpy(d_img_src, m.ptr(), (m.rows) * (m.cols), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_img_compressed, img_compressed.data, (img_compressed.rows) * (img_compressed.cols) * sizeof(double), cudaMemcpyHostToDevice);
	//cv::cuda::GpuMat src;
	//src.upload(src);
	double* f;
	cudaMalloc((void**)&f, sizeof(double) * 1);
	cudaMemcpy(f, sd, sizeof(double) * 1, cudaMemcpyHostToDevice);
	sigma << <1, 1 >> > (d_img_src, i, j, block_size, f);
	cudaMemcpy(sd, f, sizeof(double) * 1, cudaMemcpyDeviceToHost);
	cudaFree(f);
	cudaFree(d_img_src);
	//cudaFree(d_img_compressed);
	return true;
}

//double CGPUACC::getCov(Mat& m1, Mat& m2, int i, int j, int block_size, double* sd_ro) {
//	double* f;
//	cudaMalloc((void**)&f, sizeof(double) * 1);
//	cudaMemcpy(f, sd_ro, sizeof(double) * 1, cudaMemcpyHostToDevice);
//	cov << <1, 1 >> > (m1,m2, i, j, block_size, sd_ro);
//	cudaMemcpy(sd_ro, f, sizeof(int) * 1, cudaMemcpyDeviceToHost);
//	cudaFree(f);
//	return true;
//}
double CGPUACC::getSSIM(Mat& img_src, Mat& img_compressed, int block_size, double* ssim) {
	// SIGMA, COV -> Mat->double ->global__CUDA
	// 
	

	

	int nbBlockPerHeight = img_src.rows / block_size;
	int nbBlockPerWidth = img_src.cols / block_size;
	for (int k = 0; k < nbBlockPerHeight; k++)
	{
		for (int l = 0; l < nbBlockPerWidth; l++)
		{
			int m = k * block_size;
			int n = l * block_size;
			double* osd;
			double* rsd;
			double* sd_ro;
			double avg_o = mean(img_src(Range(k, k + block_size), Range(l, l + block_size)))[0];
			double avg_r = mean(img_compressed(Range(k, k + block_size), Range(l, l + block_size)))[0];

			//sigma(img_src, m, n, block_size, osd);
			//sigma(img_compressed, m, n, block_size, rsd);

			double sigma_o = *osd;
			double sigma_r = *rsd;

			//cov(img_src, img_compressed, m, n, block_size, sd_ro);
			//cov << <1, 1 >> > (img_src, img_compressed, m, n, block_size, sd_ro);
			double sigma_ro = *sd_ro;

			*ssim += ((2 * avg_o * avg_r + C1) * (2 * sigma_ro + C2)) / ((avg_o * avg_o + avg_r * avg_r + C1) * (sigma_o * sigma_o + sigma_r * sigma_r + C2));

		}
	}

	*ssim /= nbBlockPerHeight * nbBlockPerWidth;

	
	double* f;
	cudaMalloc((void**)&f, sizeof(double) * 1);
	cudaMemcpy(f, ssim, sizeof(double) * 1, cudaMemcpyHostToDevice);
	//f_ssim << <1, 1 >> > (img_src, img_compressed, block_size, ssim);
	cudaMemcpy(ssim, f, sizeof(int) * 1, cudaMemcpyDeviceToHost);
	cudaFree(f);
	return true;

}
