#include "ESMNvresizerCore.h"
#include "resizer.h"

ESMNvresizer::Core::Core(void)
	: _initialized(FALSE)
	, _cuContext(NULL)
	, _cuPitch(0)
	, _cuFrame(NULL)
{

}

ESMNvresizer::Core::~Core(void)
{

}

BOOL ESMNvresizer::Core::IsInitialized(void)
{
	return _initialized;
}

int ESMNvresizer::Core::Initialize(ESMNvresizer::CONTEXT_T * ctx)
{
	_context = ctx;
	int ngpu = 0;
	CUresult cret = ::cuInit(0, __CUDA_API_VERSION, NULL);
	cret = ::cuDeviceGetCount(&ngpu);
	if((_context->deviceIndex < 0) || (_context->deviceIndex >= ngpu))
		return -1;
	CUdevice cuDevice;
	cret = ::cuDeviceGet(&cuDevice, _context->deviceIndex);
	cret = ::cuCtxCreate(&_cuContext, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice);

	if(_context->colorspace == ESMNvresizer::COLORSPACE_T::BGRA)
	{
		::cuCtxPushCurrent(_cuContext);
		::cuMemAllocPitch((CUdeviceptr*)&_cuFrame, &_cuPitch, 4 * _context->outputWidth, _context->outputHeight, 16);
		::cuCtxPopCurrent(NULL);
	}
	else
	{
		::cuCtxPushCurrent(_cuContext);
		::cuMemAllocPitch((CUdeviceptr*)&_cuFrame, &_cuPitch, _context->outputWidth, (_context->outputHeight>>1) * 3, 16);
		::cuCtxPopCurrent(NULL);
	}
	_initialized = TRUE;

	return 0;
}

int ESMNvresizer::Core::Release(void)
{
	::cuCtxPushCurrent(_cuContext);
	::cuMemFree((CUdeviceptr)(_cuFrame));
	::cuCtxPopCurrent(NULL);
	_initialized = FALSE;

	return 0;
}

int ESMNvresizer::Core::Resize(unsigned char * input, int inputPitch, unsigned char ** output, int & outputPitch)
{
	ESMNvresizer::CUDAResizer::resize_yv12(_cuFrame, _cuPitch, _context->outputWidth, _context->outputHeight, input, inputPitch, _context->inputWidth, _context->inputHeight);
	*output = _cuFrame;
	outputPitch = _cuPitch;

	return 0;
}