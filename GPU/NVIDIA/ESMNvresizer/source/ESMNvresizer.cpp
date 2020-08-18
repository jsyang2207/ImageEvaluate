#include "ESMNvresizer.h"
#include "ESMNvresizerCore.h"

ESMNvresizer::ESMNvresizer(void)
{
	_core = new ESMNvresizer::Core();
}

ESMNvresizer::~ESMNvresizer(void)
{
	if(_core)
	{
		delete _core;
		_core = NULL;
	}
}

BOOL ESMNvresizer::IsInitialized(void)
{
	return _core->IsInitialized();
}

int ESMNvresizer::Initialize(ESMNvresizer::CONTEXT_T * ctx)
{
	return _core->Initialize(ctx);
}

int ESMNvresizer::Release(void)
{
	return _core->Release();
}

int ESMNvresizer::Resize(unsigned char * input, int inputPitch, unsigned char ** output, int & outputPitch)
{
	return _core->Resize(input, inputPitch, output, outputPitch);
}