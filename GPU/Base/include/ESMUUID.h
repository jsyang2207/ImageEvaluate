#ifndef _ESMUUID_H_
#define _ESMUUID_H_

#include <winsock2.h>
#include <windows.h>

#include <string>

#define UUID_LENGTH 16

class ESMUUID : GUID
{
private:
	std::string _uuid;

public:
	ESMUUID(void)
	{
		Data1 = 0;
		Data2 = 0;
		Data3 = 0;
		ZeroMemory(Data4, 8);
	}

	ESMUUID(uint8_t * puuid, int32_t size)
	{
		UUID uuid;

		ESMUUID::ESMUUID();

		if (size > 16)
			size = 16;

		memcpy(&uuid, puuid, size);

		Data1 = uuid.Data1;
		Data2 = uuid.Data2;
		Data3 = uuid.Data3;
		memcpy(Data4, uuid.Data4, 8);
	}

	ESMUUID(UUID & uuid)
	{
		Data1 = uuid.Data1;
		Data2 = uuid.Data2;
		Data3 = uuid.Data3;
		memcpy(Data4, uuid.Data4, 8);
	}

	ESMUUID(std::string & str)
	{
		std::string uuidstring = str;

		UUID uuid;

		UuidFromStringA((RPC_CSTR)uuidstring.c_str(), &uuid);

		Data1 = uuid.Data1;
		Data2 = uuid.Data2;
		Data3 = uuid.Data3;
		memcpy(Data4, uuid.Data4, 8);
	}

	~ESMUUID(void)
	{

	}

	UUID		create(void)
	{
		RPC_STATUS result = UuidCreate(this);
		return *this;
	}

	std::string to_string(void)
	{
		char buf[MAX_PATH] = { 0 };

		sprintf_s(buf, MAX_PATH, "%08lX-%04hX-%04hX-%02hhX%02hhX-%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX",
			Data1, Data2, Data3,
			Data4[0], Data4[1], Data4[2], Data4[3],
			Data4[4], Data4[5], Data4[6], Data4[7]);

		_uuid = buf;
		return _uuid;
	}

	std::string to_string_ntoh(void)
	{
		UUID hUuid = ntoh();

		std::string ret_uuid;

		RPC_CSTR pUuid = NULL;
		if (::UuidToStringA(&hUuid, (RPC_CSTR*)&pUuid) == RPC_S_OK)
		{
			ret_uuid = (char *)pUuid;
			::RpcStringFreeA(&pUuid);
		}
		std::transform(ret_uuid.begin(), ret_uuid.end(), ret_uuid.begin(), toupper);

		return ret_uuid;
	}

	char *		c_str(void)
	{
		to_string();
		return (char *)_uuid.c_str();
	}

	UUID &		get(void) 
	{
		return *this;
	}

	LPGUID		ptr(void) 
	{
		return this;
	}

	UUID hton()
	{
		char buf[255] = { 0 };
		UUID ret;

		ret.Data1 = htonl(Data1);
		ret.Data2 = htons(Data2);
		ret.Data3 = htons(Data3);
		memcpy(ret.Data4, Data4, 8);

		return ret;
	}

	UUID ntoh()
	{
		char buf[MAX_PATH] = { 0 };
		UUID ret;

		ret.Data1 = ntohl(Data1);
		ret.Data2 = ntohs(Data2);
		ret.Data3 = ntohs(Data3);
		memcpy(ret.Data4, Data4, 8);

		return ret;
	}

	ESMUUID & operator=(const UUID & rh)
	{
		Data1 = rh.Data1;
		Data2 = rh.Data2;
		Data3 = rh.Data3;
		memcpy(Data4, rh.Data4, 8);
		return *this;
	}

	ESMUUID & operator=(const std::string &  rh)
	{
		UUID uuid;

		std::string uuid_string = rh;

		UuidFromStringA((RPC_CSTR)uuid_string.c_str(), &uuid);

		Data1 = uuid.Data1;
		Data2 = uuid.Data2;
		Data3 = uuid.Data3;
		memcpy(Data4, uuid.Data4, 8);
		return *this;
	}

	ESMUUID & operator=(const char * rh)
	{
		unsigned char feUuid[] = { 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xfe };

		UUID uuid;

		if (rh ==(char *) -1) {
			memcpy(&uuid, feUuid, UUID_LENGTH);
			*this = uuid;
			return *this;
		}
		memcpy(&uuid, rh, UUID_LENGTH);

		Data1 = uuid.Data1;
		Data2 = uuid.Data2;
		Data3 = uuid.Data3;
		memcpy(Data4, uuid.Data4, 8);

		return *this;
	}

	BOOL operator== (ESMUUID & rh)
	{
		if (rh.Data1 == this->Data1 &&
			rh.Data2 == this->Data2 &&
			rh.Data3 == this->Data3 &&
			rh.Data4[0] == this->Data4[0] &&
			rh.Data4[1] == this->Data4[1] &&
			rh.Data4[2] == this->Data4[2] &&
			rh.Data4[3] == this->Data4[3] &&
			rh.Data4[4] == this->Data4[4] &&
			rh.Data4[5] == this->Data4[5] &&
			rh.Data4[6] == this->Data4[6] &&
			rh.Data4[7] == this->Data4[7])
			return TRUE;
		else
			return FALSE;
	}
};

#endif