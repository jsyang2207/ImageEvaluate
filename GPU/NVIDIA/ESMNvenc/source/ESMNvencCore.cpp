#include "ESMNvencCore.h"
#include "ESMLocks.h"

ESMNvenc::Core::Core(void)
	: _context(nullptr)
	, _nvencContext(nullptr)
	, _isInitialized(FALSE)
	, _ESMNvenc(nullptr)
	, _format(NV_ENC_BUFFER_FORMAT_NV12)
	, _nvencBufferCount(0)
	, _dump(INVALID_HANDLE_VALUE)
{
#if defined(WITH_HEVC_DEBUG)
	_file = ::CreateFileA("F:\\test.265", GENERIC_WRITE, FILE_SHARE_WRITE, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
#endif
}

ESMNvenc::Core::~Core(void)
{
#if defined(WITH_HEVC_DEBUG)
	if (_file != NULL && _file != INVALID_HANDLE_VALUE)
	{
		::CloseHandle(_file);
		_file = INVALID_HANDLE_VALUE;
	}
#endif
}

BOOL ESMNvenc::Core::IsInitialized(void)
{
	return _isInitialized;
}

int ESMNvenc::Core::Initialize(ESMNvenc::CONTEXT_T * ctx)
{
	_context = ctx;
	NVENCSTATUS status = NV_ENC_SUCCESS;

	do
	{
		status = InitializeCuda(_context->deviceIndex);
		if (status != NV_ENC_SUCCESS)
		{
			ReleaseCuda();
			break;
		}
		status = InitializeESMNvenc(_nvencContext, NV_ENC_DEVICE_TYPE_CUDA);
		if (status != NV_ENC_SUCCESS)
		{
			ReleaseESMNvenc();
			ReleaseCuda();
			break;
		}

		NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
		NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
		memset(&initializeParams, 0x00, sizeof(NV_ENC_INITIALIZE_PARAMS));
		memset(&encodeConfig, 0x00, sizeof(NV_ENC_CONFIG));

		initializeParams.encodeConfig = &encodeConfig;
		initializeParams.encodeConfig->version = NV_ENC_CONFIG_VER;
		initializeParams.version = NV_ENC_INITIALIZE_PARAMS_VER;

		if (_context->codec == ESMNvenc::VIDEO_CODEC_T::AVC)
			initializeParams.encodeGUID = NV_ENC_CODEC_H264_GUID;
		else if (_context->codec == ESMNvenc::VIDEO_CODEC_T::HEVC)
			initializeParams.encodeGUID = NV_ENC_CODEC_HEVC_GUID;

		initializeParams.presetGUID = NV_ENC_PRESET_LOW_LATENCY_HP_GUID;
		initializeParams.encodeWidth = _context->width;
		initializeParams.encodeHeight = _context->height;
		initializeParams.darWidth = _context->width;
		initializeParams.darHeight = _context->height;
		initializeParams.maxEncodeWidth = _context->width;
		initializeParams.maxEncodeHeight = _context->height;
		initializeParams.frameRateNum = _context->fps;
		initializeParams.frameRateDen = 1;
		initializeParams.enableEncodeAsync = 0;//NVGetCapability(initializeParams.encodeGUID, NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT);
		initializeParams.enablePTD = 1;
		initializeParams.reportSliceOffsets = 0;
		initializeParams.enableSubFrameWrite = 0;

		NV_ENC_PRESET_CONFIG presetConfig = { NV_ENC_PRESET_CONFIG_VER, {NV_ENC_CONFIG_VER} };
		status = _nvenc.nvEncGetEncodePresetConfig(_ESMNvenc, initializeParams.encodeGUID, initializeParams.presetGUID, &presetConfig);
		if (status != NV_ENC_SUCCESS)
		{
			ReleaseESMNvenc();
			ReleaseCuda();
			break;
		}

		memmove(initializeParams.encodeConfig, &presetConfig.presetCfg, sizeof(NV_ENC_CONFIG));
		switch(_context->codec)
		{
		case ESMNvenc::VIDEO_CODEC_T::AVC:
			switch (_context->profile)
			{
			case ESMNvenc::AVC_PROFILE_T::BP:
				initializeParams.encodeConfig->profileGUID = NV_ENC_H264_PROFILE_BASELINE_GUID;
				break;
			case ESMNvenc::AVC_PROFILE_T::HP:
				initializeParams.encodeConfig->profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;
				break;
			case ESMNvenc::AVC_PROFILE_T::MP:
				initializeParams.encodeConfig->profileGUID = NV_ENC_H264_PROFILE_MAIN_GUID;
				break;
			}
			break;
		case ESMNvenc::VIDEO_CODEC_T::HEVC:
			switch(_context->profile)
			{
			case ESMNvenc::HEVC_PROFILE_T::DP :
				initializeParams.encodeConfig->profileGUID = NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID;
				break;
			case ESMNvenc::HEVC_PROFILE_T::MP :
				initializeParams.encodeConfig->profileGUID = NV_ENC_HEVC_PROFILE_MAIN_GUID;
				break;
			}
			break;
		}

		if (_context->gop <= 0)
			initializeParams.encodeConfig->gopLength = NVENC_INFINITE_GOPLENGTH;
		else
			initializeParams.encodeConfig->gopLength = _context->gop;

		initializeParams.encodeConfig->frameIntervalP = 1;
		initializeParams.encodeConfig->frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
		initializeParams.encodeConfig->mvPrecision = NV_ENC_MV_PRECISION_QUARTER_PEL;
		initializeParams.encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
		initializeParams.encodeConfig->rcParams.averageBitRate = _context->bitrate;
		initializeParams.encodeConfig->rcParams.maxBitRate = _context->bitrate * 1.5;
		//initializeParams.encodeConfig->rcParams.vbvBufferSize = 8000000;
		//initializeParams.encodeConfig->rcParams.vbvInitialDelay = initializeParams.encodeConfig->rcParams.vbvBufferSize * 9 / 10;

		initializeParams.encodeConfig->rcParams.constQP.qpInterP = initializeParams.presetGUID == NV_ENC_PRESET_LOSSLESS_HP_GUID ? 0 : 28;
		initializeParams.encodeConfig->rcParams.constQP.qpIntra = initializeParams.presetGUID == NV_ENC_PRESET_LOSSLESS_HP_GUID ? 0 : 28;


		if (initializeParams.encodeGUID == NV_ENC_CODEC_H264_GUID)
		{
			//initializeParams.encodeConfig->encodeCodecConfig.h264Config.enableIntraRefresh = 1;
			initializeParams.encodeConfig->encodeCodecConfig.h264Config.chromaFormatIDC = 1;
			initializeParams.encodeConfig->encodeCodecConfig.h264Config.idrPeriod = initializeParams.encodeConfig->gopLength;
			//initializeParams.encodeConfig->encodeCodecConfig.h264Config.maxNumRefFrames = 16;
			initializeParams.encodeConfig->encodeCodecConfig.h264Config.adaptiveTransformMode = NV_ENC_H264_ADAPTIVE_TRANSFORM_ENABLE;
			initializeParams.encodeConfig->encodeCodecConfig.h264Config.fmoMode = NV_ENC_H264_FMO_AUTOSELECT;// NV_ENC_H264_FMO_DISABLE;
			initializeParams.encodeConfig->encodeCodecConfig.h264Config.bdirectMode = NV_ENC_H264_BDIRECT_MODE_AUTOSELECT;//nvenc_initialize_param.encodeConfig->frameIntervalP > 1 ? NV_ENC_H264_BDIRECT_MODE_TEMPORAL : NV_ENC_H264_BDIRECT_MODE_DISABLE;
			initializeParams.encodeConfig->encodeCodecConfig.h264Config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_AUTOSELECT;
			initializeParams.encodeConfig->encodeCodecConfig.h264Config.repeatSPSPPS = 1;
			initializeParams.encodeConfig->encodeCodecConfig.h264Config.sliceMode = 3;
			initializeParams.encodeConfig->encodeCodecConfig.h264Config.sliceModeData = 1;
			initializeParams.encodeConfig->encodeCodecConfig.h264Config.level = NV_ENC_LEVEL_H264_52;
		}
		else if (initializeParams.encodeGUID == NV_ENC_CODEC_HEVC_GUID)
		{
			initializeParams.encodeConfig->encodeCodecConfig.hevcConfig.chromaFormatIDC = 1;
			initializeParams.encodeConfig->encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 = 0; //NV12, 10BIT => 2
			initializeParams.encodeConfig->encodeCodecConfig.hevcConfig.idrPeriod = initializeParams.encodeConfig->gopLength;
			initializeParams.encodeConfig->encodeCodecConfig.hevcConfig.repeatSPSPPS = 1;
			initializeParams.encodeConfig->encodeCodecConfig.hevcConfig.sliceMode = 3;
			initializeParams.encodeConfig->encodeCodecConfig.hevcConfig.sliceModeData = 1;
			
			//initializeParams.encodeConfig->encodeCodecConfig.hevcConfig.maxTemporalLayersMinus1 = 1;
			initializeParams.encodeConfig->encodeCodecConfig.hevcConfig.level = NV_ENC_LEVEL_HEVC_51;
			initializeParams.encodeConfig->encodeCodecConfig.hevcConfig.tier = NV_ENC_TIER_HEVC_MAIN;
		}

		status = _nvenc.nvEncInitializeEncoder(_ESMNvenc, &initializeParams);
		if (status != NV_ENC_SUCCESS)
		{
			ReleaseESMNvenc();
			ReleaseCuda();
			break;
		}

		memset(_extradata, 0x00, sizeof(_extradata));
		NV_ENC_SEQUENCE_PARAM_PAYLOAD seqParamPayload;
		memset(&seqParamPayload, 0x00, sizeof(NV_ENC_SEQUENCE_PARAM_PAYLOAD));
		seqParamPayload.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
		seqParamPayload.inBufferSize = sizeof(_extradata);
		seqParamPayload.spsppsBuffer = _extradata;
		seqParamPayload.outSPSPPSPayloadSize = (uint32_t*)&_extradataSize;
		status = _nvenc.nvEncGetSequenceParams(_ESMNvenc, &seqParamPayload);

		_nvencBufferCount = 2;
		//_nvencBufferCount += initializeParams.encodeConfig->rcParams.lookaheadDepth;

		switch(_context->colorspace)
		{
		case ESMNvenc::COLORSPACE_T::NV12 :
			_format = NV_ENC_BUFFER_FORMAT_NV12;
			break;
		case ESMNvenc::COLORSPACE_T::YV12 :
			_format = NV_ENC_BUFFER_FORMAT_YV12;
			break;
		default:
			_format = NV_ENC_BUFFER_FORMAT_NV12;
			break;
		}
		status = AllocateBuffers(_context->width, _context->height, initializeParams.enableEncodeAsync ? TRUE : FALSE, _format);

	} while(0);

	_isInitialized = TRUE;
	return status;
}

int ESMNvenc::Core::Release(void)
{
	FlushEncoder();

	if (ReleaseBuffers() != NV_ENC_SUCCESS)
		return ESMNvenc::ERR_CODE_T::GENERIC_FAIL;

	if (ReleaseESMNvenc() != NV_ENC_SUCCESS)
		return ESMNvenc::ERR_CODE_T::GENERIC_FAIL;

	if (_context)
	{
		if (ReleaseCuda() != NV_ENC_SUCCESS)
			return ESMNvenc::ERR_CODE_T::GENERIC_FAIL;
	}
	_isInitialized = FALSE;
	return ESMNvenc::ERR_CODE_T::SUCCESS;
}

int ESMNvenc::Core::Encode(void * input, int inputStride, long long timestamp, unsigned char * bitstream, int bitstreamCapacity, int & bitstreamSize, long long & bitstreamTimestamp)
{
	ESMNvenc::ENTITY_T output;
	output.data = bitstream;
	output.dataCapacity = bitstreamCapacity;
	output.dataSize = 0;

	int status = Encode(input, inputStride, timestamp, &output);
	if(status != ESMNvenc::ERR_CODE_T::SUCCESS)
		return status;

	bitstreamSize = output.dataSize;
	timestamp = output.timestamp;

	return ESMNvenc::ERR_CODE_T::SUCCESS;
}

unsigned char * ESMNvenc::Core::GetExtradata(int & size)
{
	size = _extradataSize;
	return _extradata;
}

int ESMNvenc::Core::Encode(void * yuv, int yuvStride, long long timestamp, ESMNvenc::ENTITY_T * output)
{
	int status = ESMNvenc::ERR_CODE_T::SUCCESS;
	ESMNvenc::ENTITY_T input;
	input.data = (void*)yuv;
	input.dataPitch = int(yuvStride);
	input.timestamp = timestamp;
	status = Encode(&input, output);

	return ESMNvenc::ERR_CODE_T::SUCCESS;
}

int ESMNvenc::Core::Encode(ESMNvenc::ENTITY_T * input, ESMNvenc::ENTITY_T * output)
{
	NVENCSTATUS status = NV_ENC_SUCCESS;
	ESMNvenc::Core::BUFFER_T * nvencBuffer = _nvencBufferQueue.GetAvailable();
	if (nvencBuffer)
	{
		status = NVEncodeFrame(nvencBuffer, input);
	}
	
	nvencBuffer = _nvencBufferQueue.GetPending();
	while (nvencBuffer)
	{
		ProcessOutput(nvencBuffer, output);
		if (nvencBuffer)
		{
			if (nvencBuffer->input.inputPtr)
			{
				status = _nvenc.nvEncUnmapInputResource(_ESMNvenc, nvencBuffer->input.inputPtr);
				nvencBuffer->input.inputPtr = nullptr;
			}
		}
		nvencBuffer = _nvencBufferQueue.GetPending();
	}
	
	if (status != NV_ENC_SUCCESS)
		return ESMNvenc::ERR_CODE_T::GENERIC_FAIL;
	else
		return ESMNvenc::ERR_CODE_T::SUCCESS;
}

NVENCSTATUS ESMNvenc::Core::InitializeCuda(int deviceId)
{
	CUresult	result;
	CUdevice	device;
	CUcontext	current_context;
	int		deviceCount = 0;
	int		SMminor = 0, SMmajor = 0;

	typedef HMODULE CUDADRIVER;
	CUDADRIVER driver = 0;
	result = ::cuInit(0, __CUDA_API_VERSION, NULL);
	if (result != CUDA_SUCCESS)
	{
		return NV_ENC_ERR_NO_ENCODE_DEVICE;
	}

	result = ::cuDeviceGetCount(&deviceCount);
	if (result != CUDA_SUCCESS)
	{
		return NV_ENC_ERR_NO_ENCODE_DEVICE;
	}

	// If dev is negative value, we clamp to 0
	if ((int)deviceId < 0)
		deviceId = 0;

	if (deviceId >(int)deviceCount - 1)
	{
		return NV_ENC_ERR_INVALID_ENCODERDEVICE;
	}

	result = ::cuDeviceGet(&device, deviceId);
	if (result != CUDA_SUCCESS)
	{
		return NV_ENC_ERR_NO_ENCODE_DEVICE;
	}

	/*
	result = ::cuDeviceComputeCapability(&SMmajor, &SMminor, deviceId);
	if (result != CUDA_SUCCESS)
	{
		return NV_ENC_ERR_NO_ENCODE_DEVICE;
	}

	if (((SMmajor << 4) + SMminor) < 0x30)
	{
		return NV_ENC_ERR_NO_ENCODE_DEVICE;
	}
	*/

	result = ::cuCtxCreate((CUcontext*)(&_nvencContext), 0, device);
	if (result != CUDA_SUCCESS)
	{
		return NV_ENC_ERR_NO_ENCODE_DEVICE;
	}

	result = ::cuCtxPopCurrent(&current_context);
	if (result != CUDA_SUCCESS)
	{
		return NV_ENC_ERR_NO_ENCODE_DEVICE;
	}
	return NV_ENC_SUCCESS;
}

NVENCSTATUS ESMNvenc::Core::ReleaseCuda(void)
{
	CUresult result = ::cuCtxDestroy((CUcontext)_nvencContext);
	if (result != CUDA_SUCCESS)
		return NV_ENC_ERR_GENERIC;
	return NV_ENC_SUCCESS;
}

NVENCSTATUS	ESMNvenc::Core::InitializeESMNvenc(void * device, NV_ENC_DEVICE_TYPE type)
{
	NVENCSTATUS status = NV_ENC_SUCCESS;

	_nvencInstance = ::LoadLibrary(TEXT("nvEncodeAPI64.dll"));

	typedef NVENCSTATUS(NVENCAPI *NvEncodeAPIGetMaxSupportedVersion_Type)(uint32_t*);
	NvEncodeAPIGetMaxSupportedVersion_Type NvEncodeAPIGetMaxSupportedVersion = (NvEncodeAPIGetMaxSupportedVersion_Type)GetProcAddress(_nvencInstance, "NvEncodeAPIGetMaxSupportedVersion");
	uint32_t version = 0;
	uint32_t currentVersion = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
	status = NvEncodeAPIGetMaxSupportedVersion(&version);
	if (status != NV_ENC_SUCCESS)
		return status;
	if (currentVersion > version)
		return NV_ENC_ERR_GENERIC;

	typedef NVENCSTATUS(NVENCAPI *NvEncodeAPICreateInstance_Type)(NV_ENCODE_API_FUNCTION_LIST*);
	NvEncodeAPICreateInstance_Type NvEncodeAPICreateInstance = (NvEncodeAPICreateInstance_Type)GetProcAddress(_nvencInstance, "NvEncodeAPICreateInstance");
	_nvenc.version = NV_ENCODE_API_FUNCTION_LIST_VER;
	status = NvEncodeAPICreateInstance(&_nvenc);
	if (status != NV_ENC_SUCCESS)
		return status;
	
	if (!_nvenc.nvEncOpenEncodeSession)
	{
		_nvencBufferCount = 0;
		return NV_ENC_ERR_NO_ENCODE_DEVICE;
	}

	NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encodeSessionExParam = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
	encodeSessionExParam.device = device;
	encodeSessionExParam.deviceType = type;
	encodeSessionExParam.apiVersion = NVENCAPI_VERSION;
	status = _nvenc.nvEncOpenEncodeSessionEx(&encodeSessionExParam, &_ESMNvenc);
	return NV_ENC_SUCCESS;
}

NVENCSTATUS ESMNvenc::Core::ReleaseESMNvenc(void)
{
	if (!_ESMNvenc)
		return NV_ENC_ERR_GENERIC;

	_nvenc.nvEncDestroyEncoder(_ESMNvenc);
	_ESMNvenc = nullptr;

	return NV_ENC_SUCCESS;
}

NVENCSTATUS ESMNvenc::Core::AllocateBuffers(int width, int height, BOOL encodeAsync, NV_ENC_BUFFER_FORMAT format)
{
	NVENCSTATUS status = NV_ENC_SUCCESS;
	_nvencBufferQueue.Initialize(_nvencBuffer, _nvencBufferCount);

	for (int i = 0; i < _nvencBufferCount; i++)
	{
		::cuCtxPushCurrent((CUcontext)_nvencContext);
		size_t pitch = 0;
		CUdeviceptr dptr;
		int chroma_height = NVGetNumberChromaPlanes(format) * NVGetChromaHeight(format, height);
		if((format==NV_ENC_BUFFER_FORMAT_YV12) || (format==NV_ENC_BUFFER_FORMAT_IYUV))
			chroma_height = NVGetChromaHeight(format, height);
		::cuMemAllocPitch(&dptr, &pitch, NVGetWidthInBytes(format, width), height + chroma_height, 16);
		::cuCtxPopCurrent(NULL);
			
		std::vector<int> chromaOffsets;
		NVGetChromaSubplaneOffsets(format, pitch, height, chromaOffsets);
		_nvencBuffer[i].input.width = width;
		_nvencBuffer[i].input.height = height;
		_nvencBuffer[i].input.registeredPtr = NVRegisterResource((void*)dptr, NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR, width, height, int(pitch), format);
		_nvencBuffer[i].input.dptr = (void*)dptr;
		_nvencBuffer[i].input.chromaOffsets[0] = 0;
		_nvencBuffer[i].input.chromaOffsets[1] = 0;
		for (int ch = 0; ch < chromaOffsets.size(); ch++)
		{
			_nvencBuffer[i].input.chromaOffsets[ch] = chromaOffsets[ch];
		}
		_nvencBuffer[i].input.nChromaPlanes = NVGetNumberChromaPlanes(format);
		_nvencBuffer[i].input.pitch = int(pitch);
		_nvencBuffer[i].input.chromaPitch = NVGetChromaPitch(format, int(pitch));
		_nvencBuffer[i].input.format = format;
		_nvencBuffer[i].input.inputPtr = nullptr;

		NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBuffer = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER };
		status = _nvenc.nvEncCreateBitstreamBuffer(_ESMNvenc, &createBitstreamBuffer);
		_nvencBuffer[i].output.buffer = createBitstreamBuffer.bitstreamBuffer;
		if (encodeAsync)
		{
			_nvencBuffer[i].output.async = TRUE;
			_nvencBuffer[i].output.asyncEvent = ::CreateEvent(NULL, FALSE, FALSE, NULL);
			NV_ENC_EVENT_PARAMS eventParams = { NV_ENC_EVENT_PARAMS_VER };
			eventParams.completionEvent = _nvencBuffer[i].output.asyncEvent;
			_nvenc.nvEncRegisterAsyncEvent(_ESMNvenc, &eventParams);
		}
		else
		{
			_nvencBuffer[i].output.async = FALSE;
			_nvencBuffer[i].output.asyncEvent = INVALID_HANDLE_VALUE;
		}
	}
	_eosEvent = ::CreateEvent(NULL, FALSE, FALSE, NULL);
	return status;
}

NVENCSTATUS ESMNvenc::Core::ReleaseBuffers(void)
{
	::CloseHandle(_eosEvent);
	_eosEvent = INVALID_HANDLE_VALUE;


	for (int i = 0; i < _nvencBufferCount; i++)
	{
		NV_ENC_EVENT_PARAMS eventParams = { NV_ENC_EVENT_PARAMS_VER };
		eventParams.completionEvent = _nvencBuffer[i].output.asyncEvent;
		_nvenc.nvEncUnregisterAsyncEvent(_ESMNvenc, &eventParams);
		::CloseHandle(_nvencBuffer[i].output.asyncEvent);
		_nvencBuffer[i].output.asyncEvent = INVALID_HANDLE_VALUE;

		_nvenc.nvEncDestroyBitstreamBuffer(_ESMNvenc, _nvencBuffer[i].output.buffer);
		_nvencBuffer[i].output.buffer = nullptr;

		if (_nvencBuffer[i].input.inputPtr)
			_nvenc.nvEncUnmapInputResource(_ESMNvenc, _nvencBuffer[i].input.inputPtr);
		_nvenc.nvEncUnregisterResource(_ESMNvenc, _nvencBuffer[i].input.registeredPtr);

		if (_nvencBuffer[i].input.dptr)
		{
			::cuCtxPushCurrent((CUcontext)_nvencContext);
			CUdeviceptr dptr = (CUdeviceptr)(_nvencBuffer[i].input.dptr);
			::cuMemFree(dptr);
			::cuCtxPopCurrent(NULL);
		}
	}

	_nvencBufferQueue.Release();
	return NV_ENC_SUCCESS;
}

NVENCSTATUS ESMNvenc::Core::FlushEncoder(void)
{
	NVENCSTATUS status = NV_ENC_SUCCESS;
	NV_ENC_PIC_PARAMS picParams = { NV_ENC_PIC_PARAMS_VER };
	picParams.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
	picParams.completionEvent = _eosEvent;
	//if(_context->codec== ESMNvenc::VIDEO_CODEC_T::HEVC)
	//	picParams.codecPicParams.hevcPicParams.temporalId = 1;
	status = _nvenc.nvEncEncodePicture(_ESMNvenc, &picParams);
	if (status != NV_ENC_SUCCESS)
		return status;

	ESMNvenc::Core::BUFFER_T * nvenc_buffer = _nvencBufferQueue.GetPending();
	while (nvenc_buffer)
	{
		ProcessOutput(nvenc_buffer, NULL, TRUE);

		if (nvenc_buffer->input.inputPtr)
		{
			status = _nvenc.nvEncUnmapInputResource(_ESMNvenc, nvenc_buffer->input.inputPtr);
			nvenc_buffer->input.inputPtr = nullptr;
		}
		nvenc_buffer = _nvencBufferQueue.GetPending();
	}

	if (::WaitForSingleObject(_eosEvent, 500) != WAIT_OBJECT_0)
		return NV_ENC_ERR_GENERIC;
	return NV_ENC_SUCCESS;
}

NVENCSTATUS ESMNvenc::Core::ProcessOutput(ESMNvenc::Core::BUFFER_T * nvencBuffer, ESMNvenc::ENTITY_T * bitstream, BOOL flush)
{
	NVENCSTATUS status = NV_ENC_SUCCESS;
	if (!nvencBuffer || !nvencBuffer->output.buffer || nvencBuffer->output.eos)
		return NV_ENC_ERR_INVALID_PARAM;


	if (nvencBuffer->output.async)
	{
		if (nvencBuffer->output.asyncEvent == NULL || nvencBuffer->output.asyncEvent == INVALID_HANDLE_VALUE)
			return NV_ENC_ERR_INVALID_PARAM;
		if (::WaitForSingleObject(nvencBuffer->output.asyncEvent, 20000) != WAIT_OBJECT_0)
			return NV_ENC_ERR_GENERIC;
	}

	NV_ENC_LOCK_BITSTREAM lockBitstreamData = { NV_ENC_LOCK_BITSTREAM_VER };
	lockBitstreamData.outputBitstream = nvencBuffer->output.buffer;
	lockBitstreamData.doNotWait = 0;

	status = _nvenc.nvEncLockBitstream(_ESMNvenc, &lockBitstreamData);
	if (status == NV_ENC_SUCCESS)
	{
		if (!flush)
		{
			__try
			{
				if (bitstream)
				{
					if (lockBitstreamData.bitstreamSizeInBytes > (unsigned int)bitstream->dataCapacity)
						bitstream->dataSize = bitstream->dataCapacity;
					else
						bitstream->dataSize = lockBitstreamData.bitstreamSizeInBytes;
					memmove(bitstream->data, lockBitstreamData.bitstreamBufferPtr, bitstream->dataSize);
					bitstream->timestamp = nvencBuffer->input.timestamp;

#if defined(WITH_HEVC_DEBUG)
					if (_file != INVALID_HANDLE_VALUE)
					{
						uint32_t bytes_written = 0;
						uint8_t* temp = (uint8_t*)(bitstream->data);
						do
						{
							uint32_t nb_write = 0;
							::WriteFile(_file, temp, bitstream->dataSize, (LPDWORD)&nb_write, 0);
							bytes_written += nb_write;
							if (bitstream->dataSize == bytes_written)
								break;
						} while (1);
					}
#endif
				}
			}
			__except (EXCEPTION_EXECUTE_HANDLER) {}
		}
		status = _nvenc.nvEncUnlockBitstream(_ESMNvenc, lockBitstreamData.outputBitstream);
	}
	return status;
}

int ESMNvenc::Core::NVGetCapability(GUID codec, NV_ENC_CAPS caps)
{
	if (!_ESMNvenc)
		return 0;
	NV_ENC_CAPS_PARAM capsParam = { NV_ENC_CAPS_PARAM_VER };
	capsParam.capsToQuery = caps;
	int v;
	_nvenc.nvEncGetEncodeCaps(_ESMNvenc, codec, &capsParam, &v);
	return v;
}

NVENCSTATUS ESMNvenc::Core::NVEncodeFrame(ESMNvenc::Core::BUFFER_T * nvencBuffer, ESMNvenc::ENTITY_T * input)
{
	NVENCSTATUS status = NV_ENC_SUCCESS;
	{
		::cuCtxPushCurrent((CUcontext)_nvencContext);

		int chromaHeight = NVGetChromaHeight(_format, _context->height);
		int height = _context->height + chromaHeight;
		int width = NVGetWidthInBytes(_format, _context->width);

		CUDA_MEMCPY2D m = {0};
		m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		m.srcDevice = (CUdeviceptr)input->data;
		m.srcPitch = input->dataPitch;

		m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		m.dstDevice = (CUdeviceptr)nvencBuffer->input.dptr;
		m.dstPitch = nvencBuffer->input.pitch;
		m.WidthInBytes = nvencBuffer->input.pitch;
		m.Height = height;
		::cuMemcpy2D(&m);

		::cuCtxPopCurrent(NULL);

		NV_ENC_MAP_INPUT_RESOURCE mapInputResource = { NV_ENC_MAP_INPUT_RESOURCE_VER };
		mapInputResource.registeredResource = nvencBuffer->input.registeredPtr;
		status = _nvenc.nvEncMapInputResource(_ESMNvenc, &mapInputResource);
		if (status != NV_ENC_SUCCESS)
			return status;

		nvencBuffer->input.inputPtr = mapInputResource.mappedResource;
		nvencBuffer->input.timestamp = input->timestamp;

		NV_ENC_PIC_PARAMS picParams = {};
		picParams.version = NV_ENC_PIC_PARAMS_VER;
		picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
		picParams.inputBuffer = nvencBuffer->input.inputPtr;
		picParams.bufferFmt = nvencBuffer->input.format;
		picParams.inputWidth = nvencBuffer->input.width;
		picParams.inputHeight = nvencBuffer->input.height;
		picParams.outputBitstream = nvencBuffer->output.buffer;
		picParams.completionEvent = nvencBuffer->output.asyncEvent;
		picParams.inputTimeStamp = input->timestamp;
		picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
		picParams.qpDeltaMap = 0;
		picParams.qpDeltaMapSize = 0;
		//if (_context->codec == ESMNvenc::VIDEO_CODEC_T::HEVC)
		//	picParams.codecPicParams.hevcPicParams.temporalId = 1;
		status = _nvenc.nvEncEncodePicture(_ESMNvenc, &picParams);
		if ((status != NV_ENC_SUCCESS) && (status != NV_ENC_ERR_NEED_MORE_INPUT))
			return status;
	}

	return NV_ENC_SUCCESS;
}

int ESMNvenc::Core::NVGetChromaPitch(const NV_ENC_BUFFER_FORMAT format, const int lumaP)
{
		switch (format)
	{
	case NV_ENC_BUFFER_FORMAT_NV12:
	case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
	case NV_ENC_BUFFER_FORMAT_YUV444:
	case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
		return lumaP;
	case NV_ENC_BUFFER_FORMAT_YV12:
	case NV_ENC_BUFFER_FORMAT_IYUV:
		return (lumaP + 1) / 2;
	case NV_ENC_BUFFER_FORMAT_ARGB:
	case NV_ENC_BUFFER_FORMAT_ARGB10:
	case NV_ENC_BUFFER_FORMAT_AYUV:
	case NV_ENC_BUFFER_FORMAT_ABGR:
	case NV_ENC_BUFFER_FORMAT_ABGR10:
		return 0;
	default:
		return lumaP;
	}
}

int ESMNvenc::Core::NVGetNumberChromaPlanes(const NV_ENC_BUFFER_FORMAT format)
{
	switch (format)
	{
	case NV_ENC_BUFFER_FORMAT_NV12:
	case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
		return 1;
	case NV_ENC_BUFFER_FORMAT_YV12:
	case NV_ENC_BUFFER_FORMAT_IYUV:
	case NV_ENC_BUFFER_FORMAT_YUV444:
	case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
		return 2;
	case NV_ENC_BUFFER_FORMAT_ARGB:
	case NV_ENC_BUFFER_FORMAT_ARGB10:
	case NV_ENC_BUFFER_FORMAT_AYUV:
	case NV_ENC_BUFFER_FORMAT_ABGR:
	case NV_ENC_BUFFER_FORMAT_ABGR10:
		return 0;
	default:
		return 1;
	}
}

int ESMNvenc::Core::NVGetChromaHeight(const NV_ENC_BUFFER_FORMAT format, const int lumaH)
{
	switch (format)
	{
	case NV_ENC_BUFFER_FORMAT_YV12:
	case NV_ENC_BUFFER_FORMAT_IYUV:
	case NV_ENC_BUFFER_FORMAT_NV12:
	case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
		return (lumaH + 1) / 2;
	case NV_ENC_BUFFER_FORMAT_YUV444:
	case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
		return lumaH;
	case NV_ENC_BUFFER_FORMAT_ARGB:
	case NV_ENC_BUFFER_FORMAT_ARGB10:
	case NV_ENC_BUFFER_FORMAT_AYUV:
	case NV_ENC_BUFFER_FORMAT_ABGR:
	case NV_ENC_BUFFER_FORMAT_ABGR10:
		return 0;
	default:
		return (lumaH + 1) / 2;
	}
}

int ESMNvenc::Core::NVGetChromaWidthInBytes(const NV_ENC_BUFFER_FORMAT format, const uint32_t lumaW)
{
    switch (format)
    {
    case NV_ENC_BUFFER_FORMAT_YV12:
    case NV_ENC_BUFFER_FORMAT_IYUV:
        return (lumaW + 1) / 2;
    case NV_ENC_BUFFER_FORMAT_NV12:
        return lumaW;
    case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
        return 2 * lumaW;
    case NV_ENC_BUFFER_FORMAT_YUV444:
        return lumaW;
    case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
        return 2 * lumaW;
    case NV_ENC_BUFFER_FORMAT_ARGB:
    case NV_ENC_BUFFER_FORMAT_ARGB10:
    case NV_ENC_BUFFER_FORMAT_AYUV:
    case NV_ENC_BUFFER_FORMAT_ABGR:
    case NV_ENC_BUFFER_FORMAT_ABGR10:
        return 0;
    default:
        return 0;
    }
}

void ESMNvenc::Core::NVGetChromaSubplaneOffsets(const NV_ENC_BUFFER_FORMAT format, const int pitch, const int height, std::vector<int> & chroma_offsets)
{
	chroma_offsets.clear();
	switch (format)
	{
	case NV_ENC_BUFFER_FORMAT_NV12:
	case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
		chroma_offsets.push_back(pitch * height);
		return;
	case NV_ENC_BUFFER_FORMAT_YV12:
	case NV_ENC_BUFFER_FORMAT_IYUV:
		chroma_offsets.push_back(pitch * height);
		chroma_offsets.push_back(chroma_offsets[0] + (NVGetChromaPitch(format, pitch) * NVGetChromaHeight(format, height)));
		return;
	case NV_ENC_BUFFER_FORMAT_YUV444:
	case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
		chroma_offsets.push_back(pitch * height);
		chroma_offsets.push_back(chroma_offsets[0] + (pitch * height));
		return;
	case NV_ENC_BUFFER_FORMAT_ARGB:
	case NV_ENC_BUFFER_FORMAT_ARGB10:
	case NV_ENC_BUFFER_FORMAT_AYUV:
	case NV_ENC_BUFFER_FORMAT_ABGR:
	case NV_ENC_BUFFER_FORMAT_ABGR10:
		return;
	default:
		return;
	}
}

int ESMNvenc::Core::NVGetWidthInBytes(const NV_ENC_BUFFER_FORMAT format, const int width)
{
	switch (format) 
	{
	case NV_ENC_BUFFER_FORMAT_NV12:
	case NV_ENC_BUFFER_FORMAT_YV12:
	case NV_ENC_BUFFER_FORMAT_IYUV:
	case NV_ENC_BUFFER_FORMAT_YUV444:
		return width;
	case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
	case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
		return width * 2;
	case NV_ENC_BUFFER_FORMAT_ARGB:
	case NV_ENC_BUFFER_FORMAT_ARGB10:
	case NV_ENC_BUFFER_FORMAT_AYUV:
	case NV_ENC_BUFFER_FORMAT_ABGR:
	case NV_ENC_BUFFER_FORMAT_ABGR10:
		return width * 4;
	default:
		return 0;
	}
}

NV_ENC_REGISTERED_PTR ESMNvenc::Core::NVRegisterResource(void * buffer, NV_ENC_INPUT_RESOURCE_TYPE type, int width, int height, int pitch, NV_ENC_BUFFER_FORMAT format)
{
	NV_ENC_REGISTER_RESOURCE registerResource = { NV_ENC_REGISTER_RESOURCE_VER };
	registerResource.resourceType = type;
	registerResource.resourceToRegister = buffer;
	registerResource.width = width;
	registerResource.height = height;
	registerResource.pitch = pitch;
	registerResource.bufferFormat = format;
	NVENCSTATUS status = _nvenc.nvEncRegisterResource(_ESMNvenc, &registerResource);
	if (status != NV_ENC_SUCCESS)
		return nullptr;
	return registerResource.registeredResource;
}

BOOL ESMNvenc::Core::ConvertWide2Multibyte(wchar_t * src, char ** dst)
{
	UINT32 len = ::WideCharToMultiByte(CP_ACP, 0, src, (INT32)wcslen(src), NULL, NULL, NULL, NULL);
	(*dst) = new char[NULL, len + 1];
	::memset((*dst), 0x00, (len + 1)*sizeof(char));
	WideCharToMultiByte(CP_ACP, 0, src, -1, (*dst), len, NULL, NULL);
	return TRUE;
}

BOOL ESMNvenc::Core::ConvertMultibyte2Wide(char * src, wchar_t ** dst)
{
	UINT32 len = ::MultiByteToWideChar(CP_ACP, 0, src, (INT32)strlen(src), NULL, NULL);
	(*dst) = SysAllocStringLen(NULL, len + 1);
	::memset((*dst), 0x00, (len + 1)*sizeof(WCHAR));
	MultiByteToWideChar(CP_ACP, 0, src, -1, (*dst), len);

	return TRUE;
}