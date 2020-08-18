#ifndef _CUDA_RESIZER_H_
#define _CUDA_RESIZER_H_

#include <cstdint>

class resizer
{
public:
	static void resize_nv12(unsigned char * dstNV12, int dstNV12Pitch, int dstNV12Width, int dstNV12Height,  unsigned char * srcNV12, int srcNV12Picth, int srcNV12Width, int srcNV12Height);
};

#endif