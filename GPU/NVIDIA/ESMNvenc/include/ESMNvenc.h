#ifndef _ESM_NVENC_H_
#define _ESM_NVENC_H_

#if defined(EXPORT_ESM_NVENC)
#define ESM_NVENC_CLASS __declspec(dllexport)
#else
#define ESM_NVENC_CLASS __declspec(dllimport)
#endif

#include <ESMBase.h>

class ESM_NVENC_CLASS ESMNvenc
	: public ESMBase
{
	class Core;
public:
	typedef struct _CONTEXT_T
	{
		int deviceIndex;
		int width;
		int height;
		int codec;
		int profile;
		int gop;
		int bitrate;
		int colorspace;
		int fps;
		_CONTEXT_T(VOID)
			: deviceIndex(0)
			, width(3840)
			, height(2160)
			, codec(ESMNvenc::VIDEO_CODEC_T::HEVC)
			, profile(ESMNvenc::HEVC_PROFILE_T::DP)
			, gop(1)
			, fps(30)
			, bitrate(70000000)
			, colorspace(ESMNvenc::COLORSPACE_T::NV12)
		{}
	} CONTEXT_T;

	typedef struct _ENTITY_T
	{
		void *		data;
		int			dataPitch;
		int			dataSize;
		int			dataCapacity;
		long long	timestamp;
		_ENTITY_T(void)
			: data(NULL)
			, dataPitch(0)
			, dataSize(0)
			, dataCapacity(0)
			, timestamp(0)
		{}

	} ENTITY_T;
	
	ESMNvenc(void);
	virtual ~ESMNvenc(void);
	
	

	BOOL	IsInitialized(void);
	int		Initialize(ESMNvenc::CONTEXT_T * ctx);
	int		Release(void);

	int		Encode(void * input, int inputStride, long long timestamp, unsigned char * bitstream, int bitstreamCapacity, int & bitstreamSize, long long & bitstreamTimestamp);

	unsigned char * GetExtradata(int & size);

private:
	ESMNvenc(const ESMNvenc & clone);

private:
	ESMNvenc::Core* _core;
};

#endif