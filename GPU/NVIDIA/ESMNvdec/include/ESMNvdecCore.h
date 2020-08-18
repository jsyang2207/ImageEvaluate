#ifndef _ESM_NVDEC_CORE_H_
#define _ESM_NVDEC_CORE_H_

#include "ESMNvdec.h"
#include <dynlink_nvcuvid.h>
#include <dynlink_cuda.h>
#include <dynlink_cudaD3D11.h>
#include <algorithm>
#include <vector>

class ESMNvdec::Core
{
public:
	Core(void);
	virtual ~Core(void);

	BOOL	IsInitialized(void);

	int		Initialize(ESMNvdec::CONTEXT_T * ctx);
	int		Release(void);

	int		Decode(unsigned char * bitstream, int bitstreamSize, long long bitstreamTimestamp, unsigned char *** nv12, int * numberOfDecoded, long long ** timetstamp);
	size_t	GetPitch(void);
	size_t	GetPitchResized(void);
	size_t	GetPitchConverted(void);
	size_t	GetPitch2(void);
private:
	int ProcessVideoSequence(CUVIDEOFORMAT * format);
	int ProcessPictureDecode(CUVIDPICPARAMS * picture);
	int ProcessPictureDisplay(CUVIDPARSERDISPINFO * display);
	int GetNumberofDecodeSurfaces(cudaVideoCodec codec, int width, int height);

	static int __stdcall ProcessVideoSequence(void * user_data, CUVIDEOFORMAT * format);
	static int __stdcall ProcessPictureDecode(void * user_data, CUVIDPICPARAMS * picture);
	static int __stdcall ProcessPictureDisplay(void * user_data, CUVIDPARSERDISPINFO * display);

private:
	Core(const ESMNvdec::Core & clone);

private:
	BOOL _initialized;
	ESMNvdec::CONTEXT_T * _context;
	CRITICAL_SECTION	_lock;
	CRITICAL_SECTION	_lock2;
	CUcontext			_cuContext;
	CUvideoctxlock		_cuCtxLock;
	CUvideoparser		_cuParser;
	CUvideodecoder		_cuDecoder;

	int						_cuWidth;
	int						_cuHeight;
	int						_cuSurfaceHeight;
	cudaVideoCodec			_cuCodec;
	cudaVideoChromaFormat	_cuChromaFormat;
	int						_cuBitdepthMinus8;
	CUVIDEOFORMAT			_cuFormat;
	size_t					_cuPitch;
	size_t					_cuPitchResized;
	size_t					_cuPitchConverted;
	size_t					_cuPitch2;

	std::vector<unsigned char*>	_vFrame;
	std::vector<unsigned char*>	_vFrameResized;
	std::vector<unsigned char*>	_vFrameConverted;
	std::vector<unsigned char*>	_vFrame2;
	std::vector<long long>		_vTimestamp;
	CRITICAL_SECTION			_frameLock;
	int							_nDecodedFrame;
	int							_ndecodedFrameReturned;
};

#endif