#include "ESMNvresizer.h"
#include <dynlink_cuda.h>

class ESMNvresizer::Core
{
public:
	Core(void);
	virtual ~Core(void);

	BOOL	IsInitialized(void);

	int		Initialize(ESMNvresizer::CONTEXT_T * ctx);
	int		Release(void);
	int		Resize(unsigned char * input, int inputPitch, unsigned char ** output, int & outputPitch);

private:
	BOOL						_initialized;
	CUcontext					_cuContext;
	ESMNvresizer::CONTEXT_T *	_context;
	size_t						_cuPitch;
	unsigned char *				_cuFrame;
};