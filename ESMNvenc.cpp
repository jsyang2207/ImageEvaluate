#include "pch.h"
#include "ESMNvenc.h"
#include "ESMNvencCore.h"

ESMNvenc::ESMNvenc(VOID)
{
	_core = new ESMNvenc::Core();
}

ESMNvenc::~ESMNvenc(VOID)
{
	if (_core)
	{
		if (_core->IsInitialized())
			_core->Release();
		delete _core;
		_core = NULL;
	}
}

BOOL ESMNvenc::IsInitialized(void)
{
	return _core->IsInitialized();
}

int ESMNvenc::Initialize(ESMNvenc::CONTEXT_T* ctx)
{
	return _core->Initialize(ctx);
}

int ESMNvenc::Release(VOID)
{
	return _core->Release();
}

int ESMNvenc::Encode(void* input, int inputStride, long long timestamp, unsigned char* bitstream, int bitstreamCapacity, int& bitstreamSize, long long& bitstreamTimestamp)
{
	return _core->Encode(input, inputStride, timestamp, bitstream, bitstreamCapacity, bitstreamSize, bitstreamTimestamp);
}

unsigned char* ESMNvenc::GetExtradata(int& size)
{
	return _core->GetExtradata(size);
}