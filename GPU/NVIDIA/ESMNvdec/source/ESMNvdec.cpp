#include "ESMNvdec.h"
#include "ESMNvdecCore.h"

ESMNvdec::ESMNvdec(void)
{
	_core = new ESMNvdec::Core();
}

ESMNvdec::~ESMNvdec(void)
{
	if(_core)
	{
		if(_core->IsInitialized())
			_core->Release();
		delete _core;
		_core = NULL;
	}
}

BOOL ESMNvdec::IsInitialized(void)
{
	return _core->IsInitialized();
}

int ESMNvdec::Initialize(ESMNvdec::CONTEXT_T * ctx)
{
	return _core->Initialize(ctx);
}

int ESMNvdec::Release(void)
{
	return _core->Release();
}

int ESMNvdec::Decode(unsigned char * bitstream, int bitstreamSize, long long bitstreamTimestamp, unsigned char *** decoded, int * numberOfDecoded, long long ** timetstamp)
{
	return _core->Decode(bitstream, bitstreamSize, bitstreamTimestamp, decoded, numberOfDecoded, timetstamp);
}

size_t ESMNvdec::GetPitch(void)
{
	return _core->GetPitch();
}

size_t ESMNvdec::GetPitchResized(void)
{
	return _core->GetPitchResized();
}

size_t ESMNvdec::GetPitchConverted(void)
{
	return _core->GetPitchConverted();
}

size_t ESMNvdec::GetPitch2(void)
{
	return _core->GetPitch2();
}