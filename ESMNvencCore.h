#ifndef _ESM_NVENC_CORE_H_
#define _ESM_NVENC_CORE_H_

#include "ESMNvenc.h"
#include "nvEncodeAPI.h"
#include "ESMNvencQueue.h"
#include <vector>
#include "dynlink_cuda.h"

class ESMNvenc::Core
{
public:
	//static const int MAX_ENCODE_QUEUE = 32;
	typedef struct _INPUT_BUFFER_T
	{
		int						width;
		int						height;
		void* dptr;
		NV_ENC_REGISTERED_PTR	registeredPtr;
		int						chromaOffsets[2];
		int						nChromaPlanes;
		int						pitch;
		int						chromaPitch;
		NV_ENC_BUFFER_FORMAT	format;
		NV_ENC_INPUT_PTR		inputPtr;
		long long				timestamp;
		_INPUT_BUFFER_T(void)
			: width(0)
			, height(0)
			, dptr(nullptr)
			, registeredPtr(nullptr)
			, nChromaPlanes(0)
			, pitch(0)
			, chromaPitch(0)
			, format(NV_ENC_BUFFER_FORMAT_NV12)
			, timestamp(0)
		{}
	} INPUT_BUFFER_T;

	typedef struct _OUTPUT_BUFFER_T
	{
		NV_ENC_OUTPUT_PTR		buffer;
		BOOL					async;
		HANDLE					asyncEvent;
		BOOL					eos;
		_OUTPUT_BUFFER_T(void)
			: buffer(nullptr)
			, asyncEvent(INVALID_HANDLE_VALUE)
			, eos(FALSE)
		{}
	} OUTPUT_BUFFER_T;

	typedef struct _BUFFER_T
	{
		ESMNvenc::Core::INPUT_BUFFER_T	input;
		ESMNvenc::Core::OUTPUT_BUFFER_T	output;
	} BUFFER_T;

	typedef struct _CIRCULAR_BUFFER_T
	{
		long long	timestamp;
		int			amount;
		_CIRCULAR_BUFFER_T* prev;
		_CIRCULAR_BUFFER_T* next;
	} CIRCULAR_BUFFER_T;

	Core(void);
	virtual ~Core(void);

	BOOL	IsInitialized(void);
	int		Initialize(ESMNvenc::CONTEXT_T* ctx);
	int		Release(void);

	int		Encode(void* input, int inputStride, long long timetstamp, unsigned char* bitstream, int bitstreamCapacity, int& bitstreamSize, long long& bitstreamTimestamp);

	unsigned char* GetExtradata(int& size);
private:
	int		Encode(void* input, int inputStride, long long timestamp, ESMNvenc::ENTITY_T* output);
	int		Encode(ESMNvenc::ENTITY_T* input, ESMNvenc::ENTITY_T* output);

private:
	NVENCSTATUS InitializeCuda(int deviceId);
	NVENCSTATUS ReleaseCuda(void);

	NVENCSTATUS	InitializeESMNvenc(void* device, NV_ENC_DEVICE_TYPE type);
	NVENCSTATUS ReleaseESMNvenc(void);

	NVENCSTATUS AllocateBuffers(int width, int height, BOOL encodeAsync, NV_ENC_BUFFER_FORMAT format);
	NVENCSTATUS ReleaseBuffers(void);
	NVENCSTATUS FlushEncoder(void);
	NVENCSTATUS ProcessOutput(ESMNvenc::Core::BUFFER_T* nvencBuffer, ESMNvenc::ENTITY_T* bitstream, BOOL flush = FALSE);
	int			NVGetCapability(GUID codec, NV_ENC_CAPS caps);
	NVENCSTATUS NVEncodeFrame(ESMNvenc::Core::BUFFER_T* nvencBuffer, ESMNvenc::ENTITY_T* input);

	int			NVGetChromaPitch(const NV_ENC_BUFFER_FORMAT format, const int lumaP);
	int			NVGetNumberChromaPlanes(const NV_ENC_BUFFER_FORMAT format);
	int			NVGetChromaHeight(const NV_ENC_BUFFER_FORMAT format, const int lumaH);
	int			NVGetChromaWidthInBytes(const NV_ENC_BUFFER_FORMAT format, const uint32_t lumaW);
	void		NVGetChromaSubplaneOffsets(const NV_ENC_BUFFER_FORMAT format, const int pitch, const int height, std::vector<int>& chroma_offsets);
	int			NVGetWidthInBytes(const NV_ENC_BUFFER_FORMAT format, const int width);

	NV_ENC_REGISTERED_PTR NVRegisterResource(void* buffer, NV_ENC_INPUT_RESOURCE_TYPE type, int width, int height, int pitch, NV_ENC_BUFFER_FORMAT format);

	BOOL		ConvertWide2Multibyte(wchar_t* src, char** dst);
	BOOL		ConvertMultibyte2Wide(char* src, wchar_t** dst);

private:
	Core(const ESMNvenc::Core& clone);

private:
	ESMNvenc::CONTEXT_T* _context;
	BOOL		_isInitialized;
	HINSTANCE	_nvencInstance;
	NV_ENCODE_API_FUNCTION_LIST _nvenc;
	int			_nvencBufferCount;
	void* _ESMNvenc;
	void* _nvencContext;

	NV_ENC_BUFFER_FORMAT _format;
	ESMNvenc::Core::BUFFER_T _nvencBuffer[32];
	ESMNvencQueue<ESMNvenc::Core::BUFFER_T> _nvencBufferQueue;

	HANDLE		_eosEvent;
	HANDLE		_dump;
	unsigned char	_extradata[100];
	int				_extradataSize;

#if defined(WITH_HEVC_DEBUG)
	HANDLE 		_file;
#endif

#if	defined(WITH_STREAMER)
	elastics::app::eic::lib::net::eicsp::server::context_t _streamerCtx;
	elastics::app::eic::lib::net::eicsp::server* _streamer;
#endif
};

#endif