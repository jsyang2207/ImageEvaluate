#ifndef _ESM_BASE_H_
#define _ESM_BASE_H_

#include <winsock2.h>
#include <windows.h>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <climits>
#include <memory>
#include <string>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

class ESMBase
{
public:
	typedef struct _ERR_CODE_T
	{
		static const int UNKNOWN = -1;
		static const int SUCCESS = 0;
		static const int GENERIC_FAIL = 1;
		static const int INVALID_PARAMETER = 2;
	} ERR_CODE_T;

	typedef struct _MEDIA_TYPE_T
	{
		static const int UNKNOWN = 0x00;
		static const int VIDEO = 0x01;
		static const int AUDIO = 0x02;
	} MEDIA_TYPE_T;

	typedef struct _VIDEO_CODEC_T
	{
		static const int UNKNOWN = -1;
		static const int AVC = 0;
		static const int HEVC = 1;
	} VIDEO_CODEC_T;

	typedef struct _COLORSPACE_T
	{
		static const int NV12 = 0;
		static const int BGRA = 1;
		static const int YV12 = 2;
		static const int I420 = 3;
	} COLORSPACE_T;

	typedef struct _AVC_PROFILE_T
	{
		static const int BP = 0;
		static const int HP = 1;
		static const int MP = 2;
	} AVC_PROFILE_T;

	typedef struct _HEVC_PROFILE_T
	{
		static const int DP = 0;
		static const int MP = 1;
	} HEVC_PROFILE_T;

	typedef struct _VIDEO_MEMORY_TYPE_T
	{
		static const int HOST = 0;
		static const int CUDA = 1;
	} VIDEO_MEMORY_TYPE_T;

	typedef struct _AUDIO_CODEC_T
	{
		static const int UNKNOWN = -1;
		static const int MP3 = 0;
		static const int AAC = 1;
		static const int AC3 = 2;
	} AUDIO_CODEC_T;

	typedef struct _AUDIO_SAMPLE_FORMAT_T
	{
		static const int UNKNOWN = -1;
		static const int FMT_U8 = 0;
		static const int FMT_S16 = 1;
		static const int FMT_S32 = 2;
		static const int FMT_FLOAT = 3;
		static const int FMT_DOUBLE = 4;
		static const int FMT_S64 = 5;
		static const int FMT_U8P = 6;
		static const int FMT_S16P = 7;
		static const int FMT_S32P = 8;
		static const int FMT_FLOATP = 9;
		static const int FMT_DOUBLEP = 10;
		static const int FMT_S64P = 11;
	} AUDIO_SAMPLE_FORMAT_T;

};

#endif
