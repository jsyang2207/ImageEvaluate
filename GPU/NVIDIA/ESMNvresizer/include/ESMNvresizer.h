#pragma once

#if defined(EXPORT_ESM_NVRESIZER)
#define ESM_NVRESIZER_CLASS __declspec(dllexport)
#else
#define ESM_NVRESIZER_CLASS __declspec(dllimport)
#endif

#include <ESMBase.h>

class ESM_NVRESIZER_CLASS ESMNvresizer
	: public ESMBase
{
	class CUDAResizer;
	class Core;
public:
	typedef struct _CONTEXT_T
	{
		int deviceIndex;
		int colorspace;
		int inputWidth;
		int inputHeight;
		int outputWidth;
		int outputHeight;
		_CONTEXT_T(VOID)
			: deviceIndex(0)
			, inputWidth(3840)
			, inputHeight(2160)
			, outputWidth(1920)
			, outputHeight(1080)
			, colorspace(ESMNvresizer::COLORSPACE_T::YV12)
		{}
	} CONTEXT_T;

	ESMNvresizer(void);
	virtual ~ESMNvresizer(void);

	BOOL	IsInitialized(void);

	int		Initialize(ESMNvresizer::CONTEXT_T * ctx);
	int		Release(void);
	int		Resize(unsigned char * input, int inputPitch, unsigned char ** output, int & outputPitch);

private:
	ESMNvresizer::Core * _core;
};