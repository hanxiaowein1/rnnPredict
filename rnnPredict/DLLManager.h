#pragma once
#ifndef _DLLMANAGER_H_
#define _DLLMANAGER_H_
#include <windows.h>
//�����ͷų��������õ���dll
class DLLManager
{
	HINSTANCE srpDll = nullptr;
	//HINSTANCE xgDll = nullptr;
public:
	DLLManager();
	~DLLManager();
	bool getState();
};

#endif