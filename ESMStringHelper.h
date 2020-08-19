#ifndef _ESM_STRING_HELPER_H_
#define _ESM_STRING_HELPER_H_

class ESMStringHelper
{
public:
	static BOOL	ConvertWide2Multibyte(const wchar_t * source, char ** destination)
	{
		UINT32 len = WideCharToMultiByte(CP_ACP, 0, source, (INT32)wcslen(source), NULL, NULL, NULL, NULL);
		(*destination) = new char[NULL, len + 1];
		::ZeroMemory((*destination), (len + 1) * sizeof(char));
		WideCharToMultiByte(CP_ACP, 0, source, -1, (*destination), len, NULL, NULL);

		return TRUE;
	}

	static BOOL	ConvertMultibyte2Wide(const char * source, wchar_t ** destination)
	{
		UINT32 len = MultiByteToWideChar(CP_ACP, 0, source, (INT32)strlen(source), NULL, NULL);
		(*destination) = SysAllocStringLen(NULL, len + 1);
		::ZeroMemory((*destination), (len + 1) * sizeof(WCHAR));
		MultiByteToWideChar(CP_ACP, 0, source, -1, (*destination), len);

		return TRUE;
	}

};

#endif