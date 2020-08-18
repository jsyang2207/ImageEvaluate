#include "ESMNvdecCore.h"
#include <ESMLocks.h>
#include "colorspace_converter.h"
#include "resizer.h"

ESMNvdec::Core::Core(void)
	: _initialized(FALSE)
	, _cuContext(NULL)
	, _cuParser(NULL)
	, _cuDecoder(NULL)
	, _cuWidth(0)
	, _cuHeight(0)
	, _cuSurfaceHeight(0)
	, _cuCodec(cudaVideoCodec_NumCodecs)
	, _cuPitch(0)
	, _cuPitchResized(0)
	, _cuPitchConverted(0)
	, _cuPitch2(0)
{
	::memset(&_cuFormat, 0x00, sizeof(_cuFormat));
	::InitializeCriticalSection(&_lock);
	::InitializeCriticalSection(&_lock2);
	::InitializeCriticalSection(&_frameLock);
}

ESMNvdec::Core::~Core(void)
{
	::DeleteCriticalSection(&_frameLock);
	::DeleteCriticalSection(&_lock2);
	::DeleteCriticalSection(&_lock);
	_initialized = FALSE;
}

BOOL ESMNvdec::Core::IsInitialized(void)
{
	return _initialized;
}

int ESMNvdec::Core::Initialize(ESMNvdec::CONTEXT_T * ctx)
{
	_context = ctx;
	int ngpu = 0;
	CUresult cret = ::cuInit(0, __CUDA_API_VERSION, NULL);
	cret = ::cuvidInit(0);
	cret = ::cuDeviceGetCount(&ngpu);
	if((_context->deviceIndex < 0) || (_context->deviceIndex >= ngpu))
		return -1;
	CUdevice cuDevice;
	cret = ::cuDeviceGet(&cuDevice, _context->deviceIndex);
	cret = ::cuCtxCreate(&_cuContext, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice);
	cret = ::cuvidCtxLockCreate(&_cuCtxLock, _cuContext);

	CUVIDPARSERPARAMS videoParserParameters = {};
	switch (_context->codec)
	{
	case ESMNvdec::VIDEO_CODEC_T::AVC :
		videoParserParameters.CodecType = cudaVideoCodec_H264;
		break;
	case ESMNvdec::VIDEO_CODEC_T::HEVC :
		videoParserParameters.CodecType = cudaVideoCodec_HEVC;
		break;
	}
	videoParserParameters.ulMaxNumDecodeSurfaces = 1;
	videoParserParameters.ulMaxDisplayDelay = 0;
	videoParserParameters.pUserData = this;
	videoParserParameters.pfnSequenceCallback = ESMNvdec::Core::ProcessVideoSequence;
	videoParserParameters.pfnDecodePicture = ESMNvdec::Core::ProcessPictureDecode;
	videoParserParameters.pfnDisplayPicture = ESMNvdec::Core::ProcessPictureDisplay;
	{
		ESMAutolock lock(&_lock);
		::cuvidCreateVideoParser(&_cuParser, &videoParserParameters);
	}
	_initialized = TRUE;
	return 0;
}

int ESMNvdec::Core::Release(void)
{
	ESMAutolock lock2(&_lock2);
	CUresult cret;
	if (_cuParser)
	{
		cret = ::cuvidDestroyVideoParser(_cuParser);
		_cuParser = NULL;
	}
	if (_cuDecoder)
	{
		ESMAutolock lock(&_lock);
		::cuCtxPushCurrent(_cuContext);
		cret = ::cuvidDestroyDecoder(_cuDecoder);
		::cuCtxPopCurrent(NULL);
		_cuDecoder = NULL;
	}

	{
		ESMAutolock lock(&_frameLock);
		std::vector<unsigned char*>::iterator iter;
		for(iter = _vFrame.begin(); iter!=_vFrame.end(); iter++)
		{
			ESMAutolock lock2(&_lock);
			::cuCtxPushCurrent(_cuContext);
			::cuMemFree((CUdeviceptr)(*iter));
			::cuCtxPopCurrent(NULL);
		}
		_vFrame.clear();

		if (_context != nullptr && ((_context->width != _cuWidth) || (_context->height != _cuHeight)))
		{
			ESMAutolock lock(&_frameLock);
			for(iter = _vFrameResized.begin(); iter!=_vFrameResized.end(); iter++)
			{
				ESMAutolock lock2(&_lock);
				::cuCtxPushCurrent(_cuContext);
				::cuMemFree((CUdeviceptr)(*iter));
				::cuCtxPopCurrent(NULL);
			}
			_vFrameResized.clear();
		}

		if (_context != nullptr && (_context->colorspace != ESMNvdec::COLORSPACE_T::NV12))
		{
			ESMAutolock lock(&_frameLock);
			for(iter = _vFrameConverted.begin(); iter!=_vFrameConverted.end(); iter++)
			{
				ESMAutolock lock2(&_lock);
				::cuCtxPushCurrent(_cuContext);
				::cuMemFree((CUdeviceptr)(*iter));
				::cuCtxPopCurrent(NULL);
			}
			_vFrameConverted.clear();
		}
		_vFrame2.clear();
	}
	::cuvidCtxLockDestroy(_cuCtxLock);
	::cuCtxDestroy(_cuContext);
	_initialized = FALSE;
	return 0;
}

int ESMNvdec::Core::Decode(unsigned char * bitstream, int bitstreamSize, long long bitstreamTimestamp, unsigned char *** nv12, int * numberOfDecoded, long long ** timetstamp)
{
	ESMAutolock lock2(&_lock2);

	if (!_cuParser)
		return -1;

	_nDecodedFrame = 0;
	CUVIDSOURCEDATAPACKET packet = { 0 };
	packet.payload = bitstream;
	packet.payload_size = bitstreamSize;
	packet.flags = CUVID_PKT_TIMESTAMP;
	packet.timestamp = 0;
	if (!bitstream || (bitstreamSize < 1))
	{
		packet.flags |= CUVID_PKT_ENDOFSTREAM;
	}
	{
		ESMAutolock lock(&_lock);
		::cuvidParseVideoData(_cuParser, &packet);
	}

	if (_nDecodedFrame > 0)
	{
		_vFrame2.clear();
		if(nv12)
		{
			int index = 0;
			ESMAutolock lock(&_frameLock);
			std::vector<unsigned char*>::iterator iter;
			for (iter = _vFrame.begin(); iter != (_vFrame.begin() + _nDecodedFrame); iter++, index++)
			{
				if(_context->colorspace==ESMNvdec::COLORSPACE_T::BGRA)
				{
					if ((_context->width != _cuWidth) || (_context->height != _cuHeight))
					{
						resizer::resize_nv12((unsigned char *)_vFrameResized[index], (int)_cuPitchResized, _context->width, _context->height, (*iter), (int)_cuPitch, _cuWidth, _cuHeight);
						converter::convert_nv12_to_bgra32((uint8_t*)_vFrameResized[index], (int)_cuPitchResized, (uint8_t*)_vFrameConverted[index], (int)_cuPitchConverted, _context->width, _context->height);
					}
					else
					{
						converter::convert_nv12_to_bgra32((*iter), (int)_cuPitch, (uint8_t*)_vFrameConverted[index], (int)_cuPitchConverted, _context->width, _context->height);
					}
					_cuPitch2 = _cuPitchConverted;
					_vFrame2.push_back(_vFrameConverted[index]);
				}
				else if(_context->colorspace==ESMNvdec::COLORSPACE_T::I420)
				{
					if ((_context->width != _cuWidth) || (_context->height != _cuHeight))
					{
						resizer::resize_nv12((unsigned char *)_vFrameResized[index], (int)_cuPitchResized, _context->width, _context->height, (*iter), (int)_cuPitch, _cuWidth, _cuHeight);
						converter::convert_nv12_to_i420((uint8_t*)_vFrameResized[index], (int)_cuPitchResized, (uint8_t*)_vFrameConverted[index], (int)_cuPitchConverted, _context->width, _context->height);
					}
					else
					{
						converter::convert_nv12_to_i420((*iter), (int)_cuPitch, (uint8_t*)_vFrameConverted[index], (int)_cuPitchConverted, _context->width, _context->height);
					}
					_cuPitch2 = _cuPitchConverted;
					_vFrame2.push_back(_vFrameConverted[index]);
				}
				else if(_context->colorspace==ESMNvdec::COLORSPACE_T::YV12)
				{
					if ((_context->width != _cuWidth) || (_context->height != _cuHeight))
					{
						resizer::resize_nv12((unsigned char *)_vFrameResized[index], (int)_cuPitchResized, _context->width, _context->height, (*iter), (int)_cuPitch, _cuWidth, _cuHeight);
						converter::convert_nv12_to_yv12((uint8_t*)_vFrameResized[index], (int)_cuPitchResized, (uint8_t*)_vFrameConverted[index], (int)_cuPitchConverted, _context->width, _context->height);
					}
					else
					{
						converter::convert_nv12_to_yv12((*iter), (int)_cuPitch, (uint8_t*)_vFrameConverted[index], (int)_cuPitchConverted, _context->width, _context->height);
					}
					_cuPitch2 = _cuPitchConverted;
					_vFrame2.push_back(_vFrameConverted[index]);
				}
				else
				{
					if ((_context->width != _cuWidth) || (_context->height != _cuHeight))
					{
						resizer::resize_nv12((unsigned char *)_vFrameResized[index], (int)_cuPitchResized, _context->width, _context->height, (*iter), (int)_cuPitch, _cuWidth, _cuHeight);
						_cuPitch2 = _cuPitchResized;
						_vFrame2.push_back(_vFrameResized[index]);	
					}
					else
					{
						_cuPitch2 = _cuPitch;
						_vFrame2.push_back((*iter));	
					}
				}
			}
			*nv12 = &_vFrame2[0];
		}
		if(timetstamp)
		{
			*timetstamp = &_vTimestamp[0];
		}
	}
	if(numberOfDecoded)
	{
		*numberOfDecoded = _nDecodedFrame;
	}
	return 0;
}

size_t ESMNvdec::Core::GetPitch(void)
{
	return _cuPitch;
}

size_t ESMNvdec::Core::GetPitch2(void)
{
	return _cuPitch2;
}

size_t ESMNvdec::Core::GetPitchResized(void)
{
	return _cuPitchResized;
}

size_t ESMNvdec::Core::GetPitchConverted(void)
{
	return _cuPitchConverted;
}

int ESMNvdec::Core::ProcessVideoSequence(CUVIDEOFORMAT * format)
{
	int numberOfDecodeSurfaces = GetNumberofDecodeSurfaces(format->codec, format->coded_width, format->coded_height);
	if (_cuWidth && _cuHeight) 
	{
		if((format->coded_width== _cuFormat.coded_width) && (format->coded_height== _cuFormat.coded_height))
			return numberOfDecodeSurfaces;
		return numberOfDecodeSurfaces; // this error means current cuda device isn't support dynamic resolution change
	}

	_cuCodec = format->codec;
	_cuChromaFormat = format->chroma_format;
	_cuBitdepthMinus8 = format->bit_depth_luma_minus8;
	_cuFormat = *format;

	CUVIDDECODECREATEINFO videoDecodeCreateInfo = { 0 };
	videoDecodeCreateInfo.CodecType = format->codec;
	videoDecodeCreateInfo.ChromaFormat = format->chroma_format;
	videoDecodeCreateInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
	videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;// cudaVideoDeinterlaceMode_Weave;
	videoDecodeCreateInfo.ulNumOutputSurfaces = 2;
	videoDecodeCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
	videoDecodeCreateInfo.ulNumDecodeSurfaces = numberOfDecodeSurfaces;
	videoDecodeCreateInfo.vidLock = _cuCtxLock;
	videoDecodeCreateInfo.ulWidth = format->coded_width;
	videoDecodeCreateInfo.ulHeight = format->coded_height;
	videoDecodeCreateInfo.display_area.left = format->display_area.left;
	videoDecodeCreateInfo.display_area.right = format->display_area.right;
	videoDecodeCreateInfo.display_area.top = format->display_area.top;
	videoDecodeCreateInfo.ulTargetWidth = format->coded_width;
	videoDecodeCreateInfo.ulTargetHeight = format->coded_height;

	_cuWidth = format->display_area.right - format->display_area.left;
	_cuHeight = format->display_area.bottom - format->display_area.top;	
	_cuSurfaceHeight = videoDecodeCreateInfo.ulTargetHeight;

	::cuCtxPushCurrent(_cuContext);
	CUresult cret = ::cuvidCreateDecoder(&_cuDecoder, &videoDecodeCreateInfo);
	::cuCtxPopCurrent(NULL);
	return numberOfDecodeSurfaces;
}

int ESMNvdec::Core::ProcessPictureDecode(CUVIDPICPARAMS * picture)
{
	if (!_cuDecoder)
		return -1;
	::cuvidDecodePicture(_cuDecoder, picture);

	return 1;
}

int ESMNvdec::Core::ProcessPictureDisplay(CUVIDPARSERDISPINFO * display)
{
	CUVIDPROCPARAMS videoProcessingParameters = {};
	videoProcessingParameters.progressive_frame = display->progressive_frame;
	videoProcessingParameters.second_field = display->repeat_first_field + 1;
	videoProcessingParameters.top_field_first = display->top_field_first;
	videoProcessingParameters.unpaired_field = display->repeat_first_field < 0;

	CUdeviceptr dpSrcFrame = 0;
	unsigned int srcPitch = 0;
	::cuvidMapVideoFrame(_cuDecoder, display->picture_index, &dpSrcFrame, &srcPitch, &videoProcessingParameters);
	unsigned char * pDecodedFrame = NULL;
	{
		ESMAutolock lock(&_frameLock);
		if (size_t(++_nDecodedFrame) > _vFrame.size())
		{
			//_nframe_alloc++;
			unsigned char * pFrame = NULL;
			::cuCtxPushCurrent(_cuContext);
			::cuMemAllocPitch((CUdeviceptr*)&pFrame, &_cuPitch, _cuWidth * (_cuBitdepthMinus8 ? 2 : 1), (_cuHeight >> 1) * 3, 16);
			::cuCtxPopCurrent(NULL);
			_vFrame.push_back(pFrame);

			if ((_context->width != _cuWidth) || (_context->height != _cuHeight))
			{
				::cuCtxPushCurrent(_cuContext);
				::cuMemAllocPitch((CUdeviceptr*)&pFrame, &_cuPitchResized, _context->width * (_cuBitdepthMinus8 ? 2 : 1), (_context->height >> 1) * 3, 16);
				::cuCtxPopCurrent(NULL);
				_vFrameResized.push_back(pFrame);
			}
			if (_context->colorspace != ESMNvdec::COLORSPACE_T::NV12)
			{
				if(_context->colorspace == ESMNvdec::COLORSPACE_T::BGRA)
				{
					::cuCtxPushCurrent(_cuContext);
					::cuMemAllocPitch((CUdeviceptr*)&pFrame, &_cuPitchConverted, 4 * _context->width, _context->height, 16);
					::cuCtxPopCurrent(NULL);
					_vFrameConverted.push_back(pFrame);
				}
				else
				{
					size_t cuPitchConverted = 0;
					::cuCtxPushCurrent(_cuContext);
					CUresult cret = ::cuMemAllocPitch((CUdeviceptr*)&pFrame, &cuPitchConverted, _context->width, (_context->height>>1) * 3, 16);
					::cuCtxPopCurrent(NULL);

					_vFrameConverted.push_back(pFrame);
					_cuPitchConverted = cuPitchConverted;
				}
			}

			/*
			if(_context->colorspace == ESMNvdec::COLORSPACE_T::BGRA)
			{
				pFrame = NULL;
				::cuCtxPushCurrent(_cuContext);
				::cuMemAllocPitch((CUdeviceptr*)&pFrame, &_cuPitch2, _cuWidth << 2, _cuHeight, 16);
				::cuCtxPopCurrent(NULL);
				_vFrame2.push_back(pFrame);
			}
			else if(_context->colorspace == ESMNvdec::COLORSPACE_T::YV12)
			{
				pFrame = NULL;
				::cuCtxPushCurrent(_cuContext);
				::cuMemAllocPitch((CUdeviceptr*)&pFrame, &_cuPitch2, _cuWidth, (_cuHeight >> 1) * 3, 16);
				::cuCtxPopCurrent(NULL);
				_vFrame2.push_back(pFrame);
			}
			else if(_context->colorspace == ESMNvdec::COLORSPACE_T::I420)
			{
				pFrame = NULL;
				::cuCtxPushCurrent(_cuContext);
				::cuMemAllocPitch((CUdeviceptr*)&pFrame, &_cuPitch2, _cuWidth, (_cuHeight >> 1) * 3, 16);
				::cuCtxPopCurrent(NULL);
				_vFrame2.push_back(pFrame);
			}
			*/
		}
		pDecodedFrame = _vFrame[_nDecodedFrame - 1];
	}

	::cuCtxPushCurrent(_cuContext);
	CUDA_MEMCPY2D m = { 0 };
	m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	m.srcDevice = dpSrcFrame;
	m.srcPitch = srcPitch;
	m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	m.dstDevice = (CUdeviceptr)(pDecodedFrame);
	m.dstPitch = _cuPitch ? _cuPitch : _cuWidth * (_cuBitdepthMinus8 ? 2 : 1);
	m.WidthInBytes = _cuWidth * (_cuBitdepthMinus8 ? 2 : 1);
	m.Height = _cuHeight;
	::cuMemcpy2DAsync(&m, 0);

	m.srcDevice = (CUdeviceptr)((unsigned char *)dpSrcFrame + m.srcPitch * _cuSurfaceHeight);
	m.dstDevice = (CUdeviceptr)(m.dstHost = pDecodedFrame + m.dstPitch * _cuHeight);
	m.Height = _cuHeight >> 1;
	::cuMemcpy2DAsync(&m, 0);
	::cuStreamSynchronize(0);
	::cuCtxPopCurrent(NULL);

	if (int(_vTimestamp.size()) < _nDecodedFrame)
		_vTimestamp.resize(_vFrame.size());
	_vTimestamp[_nDecodedFrame - 1] = display->timestamp;

	::cuvidUnmapVideoFrame(_cuDecoder, dpSrcFrame);

	return 1;
}

int ESMNvdec::Core::GetNumberofDecodeSurfaces(cudaVideoCodec codec, int width, int height)
{
	if (codec == cudaVideoCodec_H264) 
		return 20;
	if (codec == cudaVideoCodec_HEVC) 
	{
		// ref HEVC spec: A.4.1 General tier and level limits
		// currently assuming level 6.2, 8Kx4K
		int MaxLumaPS = 35651584;
		int MaxDpbPicBuf = 6;
		int PicSizeInSamplesY = (int)(width * height);
		int MaxDpbSize;
		if (PicSizeInSamplesY <= (MaxLumaPS >> 2))
			MaxDpbSize = MaxDpbPicBuf * 4;
		else if (PicSizeInSamplesY <= (MaxLumaPS >> 1))
			MaxDpbSize = MaxDpbPicBuf * 2;
		else if (PicSizeInSamplesY <= ((3 * MaxLumaPS) >> 2))
			MaxDpbSize = (MaxDpbPicBuf * 4) / 3;
		else
			MaxDpbSize = MaxDpbPicBuf;
		return (std::min)(MaxDpbSize, 16) + 4;
	}
	return 8;
}

int ESMNvdec::Core::ProcessVideoSequence(void * user_data, CUVIDEOFORMAT * format)
{
	return (static_cast<ESMNvdec::Core*>(user_data))->ProcessVideoSequence(format);
}

int ESMNvdec::Core::ProcessPictureDecode(void * user_data, CUVIDPICPARAMS * picture)
{
	return (static_cast<ESMNvdec::Core*>(user_data))->ProcessPictureDecode(picture);
}

int ESMNvdec::Core::ProcessPictureDisplay(void * user_data, CUVIDPARSERDISPINFO * display)
{
	return (static_cast<ESMNvdec::Core*>(user_data))->ProcessPictureDisplay(display);
}