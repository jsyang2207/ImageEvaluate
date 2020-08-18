#pragma once

#include "ESMNvresizer.h"

class ESMNvresizer::CUDAResizer
{
public:
	static void resize_nv12(unsigned char * dstNV12, int dstNV12Pitch, int dstNV12Width, int dstNV12Height,  unsigned char * srcNV12, int srcNV12Picth, int srcNV12Width, int srcNV12Height);
	static void resize_yv12(unsigned char * dstYV12, int dstYV12Pitch, int dstYV12Width, int dstYV12Height,  unsigned char * srcYV12, int srcYV12Picth, int srcYV12Width, int srcYV12Height);
};