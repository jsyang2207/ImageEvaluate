#ifndef _ESM_LOCKS_H_
#define _ESM_LOCKS_H_

class ESMAutolock
{
public:
	ESMAutolock(CRITICAL_SECTION* lock)
		: _lock(lock)
	{
		::EnterCriticalSection(_lock);
	}

	~ESMAutolock(void)
	{
		::LeaveCriticalSection(_lock);
	}
private:
	CRITICAL_SECTION* _lock;
};

class ESMScopedlock
{
public:
	ESMScopedlock(HANDLE * lock)
		: _lock(lock)
	{
		if (_lock == NULL || _lock == INVALID_HANDLE_VALUE)
			return;
		::WaitForSingleObject(_lock, INFINITE);
	}

	~ESMScopedlock(void)
	{
		if (_lock == NULL || _lock == INVALID_HANDLE_VALUE)
			return;

		::SetEvent(_lock);
	}
private:
	HANDLE * _lock;
};

#endif