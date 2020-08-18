#include <cuda_runtime.h>
#include "resizer.h"
////////////////////////device////////////////////
template<class T>
__device__ static T clamp(T x, T lower, T upper)
{
	return x < lower ? lower : (x > upper ? upper : x);
}

template<typename yuv>
static __global__ void resizeNV12(cudaTextureObject_t texY, cudaTextureObject_t texUV, unsigned char *dst, unsigned char *dstUV, int pitch, int width, int height, float fxScale, float fyScale)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x,
		iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width / 2 || iy >= height / 2)
		return;

	int x = ix * 2, y = iy * 2;
	typedef decltype(yuv::x) YuvUnit;
	const int MAX = 1 << (sizeof(YuvUnit) * 8);
	
	yuv data;
	data.x = (YuvUnit)clamp((float)(tex2D<float>(texY, x / fxScale, y / fyScale) * MAX), 0.0f, 255.0f);
	data.y = (YuvUnit)clamp((float)(tex2D<float>(texY, (x + 1) / fxScale, y / fyScale) * MAX), 0.0f, 255.0f);
	//data.x = (YuvUnit)(tex2D<float>(texY, x / fxScale, y / fyScale) * MAX);
	//data.y = (YuvUnit)(tex2D<float>(texY, (x + 1) / fxScale, y / fyScale) * MAX);
	*(yuv *)(dst + y * pitch + x * sizeof(YuvUnit)) = data;
	//*(yuv *)(dst + y * pitch + x * sizeof(YuvUnit)) = yuv { (YuvUnit)(tex2D<float>(texY, x / fxScale, y / fyScale) * MAX), (YuvUnit)(tex2D<float>(texY, (x + 1) / fxScale, y / fyScale) * MAX) };
	
	y++;

	data.x = (YuvUnit)clamp((float)(tex2D<float>(texY, x / fxScale, y / fyScale) * MAX), 0.0f, 255.0f);
	data.y = (YuvUnit)clamp((float)(tex2D<float>(texY, (x + 1) / fxScale, y / fyScale) * MAX), 0.0f, 255.0f);
	//data.x = (YuvUnit)(tex2D<float>(texY, x / fxScale, y / fyScale) * MAX);
	//data.y = (YuvUnit)(tex2D<float>(texY, (x + 1) / fxScale, y / fyScale) * MAX);
	*(yuv *)(dst + y * pitch + x * sizeof(YuvUnit)) = data;
	//*(yuv *)(dst + y * pitch + x * sizeof(YuvUnit)) = yuv { (YuvUnit)(tex2D<float>(texY, x / fxScale, y / fyScale) * MAX), (YuvUnit)(tex2D<float>(texY, (x + 1) / fxScale, y / fyScale) * MAX) };
	
	float2 uv = tex2D<float2>(texUV, ix / fxScale, (height + iy) / fyScale + 0.5f);
	data.x = (YuvUnit)clamp((float)(uv.x * MAX), 0.0f, 255.0f);
	data.y = (YuvUnit)clamp((float)(uv.y * MAX), 0.0f, 255.0f);
	//data.x = (YuvUnit)(uv.x * MAX);
	//data.y = (YuvUnit)(uv.y * MAX);
	*(yuv *)(dstUV + iy * pitch + ix * 2 * sizeof(YuvUnit)) = data;
	//*(yuv *)(dstUV + iy * pitch + ix * 2 * sizeof(YuvUnit)) = yuv{ (YuvUnit)(uv.x * MAX), (YuvUnit)(uv.y * MAX) };
}

static void resizeNV12(unsigned char * dst, unsigned char * dstChroma, int dstPitch, int dstWidth, int dstHeight, unsigned char * src, int srcPitch, int srcWidth, int srcHeight)
{
	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = src;
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<decltype(uchar2::x)>();
	resDesc.res.pitch2D.width = srcWidth;
	resDesc.res.pitch2D.height = srcHeight;
	resDesc.res.pitch2D.pitchInBytes = srcPitch;

	cudaTextureDesc texDesc = {};
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;

	cudaTextureObject_t texY = 0;
	cudaCreateTextureObject(&texY, &resDesc, &texDesc, NULL);

	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar2>();
	resDesc.res.pitch2D.width = srcWidth / 2;
	resDesc.res.pitch2D.height = srcHeight * 3 / 2;

	cudaTextureObject_t texUV = 0;
	cudaCreateTextureObject(&texUV, &resDesc, &texDesc, NULL);

	resizeNV12<uchar2><<<dim3((dstWidth + 31) / 32, (dstHeight + 31) / 32), dim3(16, 16)>>>(texY, texUV, dst, dstChroma, dstPitch, dstWidth, dstHeight, 1.0f * dstWidth / srcWidth, 1.0f * dstHeight / srcHeight);

	cudaDestroyTextureObject(texY);
	cudaDestroyTextureObject(texUV);
}

/*
template<typename yuv>
static __global__ void resizeYV12(cudaTextureObject_t texY, cudaTextureObject_t texU, cudaTextureObject_t texV, unsigned char *dst, unsigned char *dstU, unsigned char *dstV, int pitch, int width, int height, float fxScale, float fyScale)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x,
		iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width / 2 || iy >= height / 2)
		return;

	int x = ix * 2, y = iy * 2;
	typedef decltype(yuv::x) YuvUnit;
	const int MAX = 1 << (sizeof(YuvUnit) * 8);
	
	yuv data;
	data.x = (YuvUnit)(tex2D<float>(texY, x / fxScale, y / fyScale) * MAX);
	data.y = (YuvUnit)(tex2D<float>(texY, (x + 1) / fxScale, y / fyScale) * MAX);
	*(yuv *)(dst + y * pitch + x * sizeof(YuvUnit)) = data;
	
	y++;
	
	data.x = (YuvUnit)(tex2D<float>(texY, x / fxScale, y / fyScale) * MAX);
	data.y = (YuvUnit)(tex2D<float>(texY, (x + 1) / fxScale, y / fyScale) * MAX);
	*(yuv *)(dst + y * pitch + x * sizeof(YuvUnit)) = data;

	float2 u = tex2D<float2>(texU, ix / fxScale, (height + iy) / fyScale + 0.5f);
	data.x = (YuvUnit)(u.x * MAX);
	data.y = (YuvUnit)(u.y * MAX);
	*(yuv *)(dstU + y * pitch + x * sizeof(YuvUnit)) = data;

	float2 v = tex2D<float2>(texV, ix / fxScale, (height + height/2 + iy) / fyScale + 0.5f);
	data.x = (YuvUnit)(v.x * MAX);
	data.y = (YuvUnit)(v.y * MAX);
	*(yuv *)(dstV + y * pitch + x * sizeof(YuvUnit)) = data;
}

static void resizeYV12(unsigned char * dst, unsigned char * dstU, unsigned char * dstV, int dstPitch, int dstWidth, int dstHeight, unsigned char * src, int srcPitch, int srcWidth, int srcHeight)
{
	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = src;
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<decltype(uchar2::x)>();
	resDesc.res.pitch2D.width = srcWidth;
	resDesc.res.pitch2D.height = srcHeight;
	resDesc.res.pitch2D.pitchInBytes = srcPitch;

	cudaTextureDesc texDesc = {};
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;

	cudaTextureObject_t texY = 0;
	cudaCreateTextureObject(&texY, &resDesc, &texDesc, NULL);

	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar2>();
	resDesc.res.pitch2D.width = srcWidth / 2;
	resDesc.res.pitch2D.height = srcHeight / 2;

	cudaTextureObject_t texU = 0;
	cudaCreateTextureObject(&texU, &resDesc, &texDesc, NULL);

	cudaTextureObject_t texV = 0;
	cudaCreateTextureObject(&texV, &resDesc, &texDesc, NULL);

	resizeYV12<uchar2><<<dim3((dstWidth + 31) / 32, (dstHeight + 31) / 32), dim3(16, 16)>>>(texY, texU, texV, dst, dstU, dstV, dstPitch, dstWidth, dstHeight, 1.0f * dstWidth / srcWidth, 1.0f * dstHeight / srcHeight);

	cudaDestroyTextureObject(texY);
	cudaDestroyTextureObject(texU);
	cudaDestroyTextureObject(texV);
}
*/

template<typename yuv>
static __global__ void resizeYV12(cudaTextureObject_t tex, unsigned char *dst, int pitch, int width, int height, float fxScale, float fyScale)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x,
		iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= width / 2 || iy >= height / 2)
		return;

	int x = ix * 2, y = iy * 2;
	typedef decltype(yuv::x) YuvUnit;
	const int MAX = 1 << (sizeof(YuvUnit) * 8);
	
	yuv data;
	data.x = (YuvUnit)clamp((tex2D<float>(tex, x / fxScale, y / fyScale) * MAX), 0.0f, 255.0f);
	data.y = (YuvUnit)clamp((tex2D<float>(tex, (x + 1) / fxScale, y / fyScale) * MAX), 0.0f, 255.0f);
	//data.x = (YuvUnit)(tex2D<float>(tex, x / fxScale, y / fyScale) * MAX);
	//data.y = (YuvUnit)(tex2D<float>(tex, (x + 1) / fxScale, y / fyScale) * MAX);
	*(yuv *)(dst + y * pitch + x * sizeof(YuvUnit)) = data;
	
	y++;
	
	data.x = (YuvUnit)clamp((tex2D<float>(tex, x / fxScale, y / fyScale) * MAX), 0.0f, 255.0f);
	data.y = (YuvUnit)clamp((tex2D<float>(tex, (x + 1) / fxScale, y / fyScale) * MAX), 0.0f, 255.0f);
	//data.x = (YuvUnit)(tex2D<float>(tex, x / fxScale, y / fyScale) * MAX);
	//data.y = (YuvUnit)(tex2D<float>(tex, (x + 1) / fxScale, y / fyScale) * MAX);
	*(yuv *)(dst + y * pitch + x * sizeof(YuvUnit)) = data;
}

static void resizeYV12(unsigned char * dst, int dstPitch, int dstWidth, int dstHeight, unsigned char * src, int srcPitch, int srcWidth, int srcHeight)
{
	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = src;
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<decltype(uchar2::x)>();
	resDesc.res.pitch2D.width = srcWidth;
	resDesc.res.pitch2D.height = srcHeight;
	resDesc.res.pitch2D.pitchInBytes = srcPitch;

	cudaTextureDesc texDesc = {};
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;

	cudaTextureObject_t tex = 0;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

	resizeYV12<uchar2><<<dim3((dstWidth + 31) / 32, (dstHeight + 31) / 32), dim3(16, 16)>>>(tex, dst, dstPitch, dstWidth, dstHeight, 1.0f * dstWidth / srcWidth, 1.0f * dstHeight / srcHeight);

	cudaDestroyTextureObject(tex);
}

void ESMNvresizer::CUDAResizer::resize_nv12(unsigned char * dstNV12, int dstNV12Pitch, int dstNV12Width, int dstNV12Height, unsigned char * srcNV12, int srcNV12Pitch, int srcNV12Width, int srcNV12Height)
{
	unsigned char * dstNV12Chroma = dstNV12 + (dstNV12Pitch * dstNV12Height);
	return resizeNV12(dstNV12, dstNV12Chroma, dstNV12Pitch, dstNV12Width, dstNV12Height, srcNV12, srcNV12Pitch, srcNV12Width, srcNV12Height);
}

void ESMNvresizer::CUDAResizer::resize_yv12(unsigned char * dstYV12, int dstYV12Pitch, int dstYV12Width, int dstYV12Height,  unsigned char * srcYV12, int srcYV12Pitch, int srcYV12Width, int srcYV12Height)
{
	int dstChromaPitch = dstYV12Pitch >> 1;
	int dstChromaWidth = dstYV12Width >> 1;
	int dstChromaHeight = dstYV12Height >> 1;

	int srcChromaPitch = srcYV12Pitch >> 1;
	int srcChromaWidth = srcYV12Width >> 1;
	int srcChromaHeight = srcYV12Height >> 1;

	unsigned char * dstU = dstYV12 + (dstYV12Pitch * dstYV12Height);
	unsigned char * dstV = dstU + (dstChromaPitch * dstChromaHeight);
	
	unsigned char * srcU = srcYV12 + (srcYV12Pitch * srcYV12Height);
	unsigned char * srcV = srcU + (srcChromaPitch * srcChromaHeight);
	
	resizeYV12(dstYV12, dstYV12Pitch, dstYV12Width, dstYV12Height, srcYV12, srcYV12Pitch, srcYV12Width, srcYV12Height);
	resizeYV12(dstU, dstChromaPitch, dstChromaWidth, dstChromaHeight, srcU, srcChromaPitch, srcChromaWidth, srcChromaHeight);
	resizeYV12(dstV, dstChromaPitch, dstChromaWidth, dstChromaHeight, srcV, srcChromaPitch, srcChromaWidth, srcChromaHeight);
}
