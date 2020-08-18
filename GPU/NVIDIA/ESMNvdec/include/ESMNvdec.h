#ifndef _ESM_NVDEC_H_
#define _ESM_NVDEC_H_

#if defined(EXPORT_ESM_NVDEC)
#define ESM_NVDEC_CLASS __declspec(dllexport)
#else
#define ESM_NVDEC_CLASS __declspec(dllimport)
#endif

#include <ESMBase.h>

class ESM_NVDEC_CLASS ESMNvdec
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
		int colorspace;
		_CONTEXT_T(VOID)
			: deviceIndex(0)
			, width(3840)
			, height(2160)
			, codec(ESMNvdec::VIDEO_CODEC_T::AVC)
			, colorspace(ESMNvdec::COLORSPACE_T::BGRA)
		{}
	} CONTEXT_T;


	ESMNvdec(void);
	virtual ~ESMNvdec(void);

	BOOL	IsInitialized(void);

	int		Initialize(ESMNvdec::CONTEXT_T * ctx);
	int		Release(void);
	int		Decode(unsigned char * bitstream, int bitstreamSize, long long bitstreamTimestamp, unsigned char *** decoded, int * numberOfDecoded, long long ** timetstamp);

	size_t	GetPitch(void);
	size_t	GetPitchResized(void);
	size_t	GetPitchConverted(void);
	size_t	GetPitch2(void);

private:
	ESMNvdec(const ESMNvdec & clone);

private:
	ESMNvdec::Core * _core;
};

#endif