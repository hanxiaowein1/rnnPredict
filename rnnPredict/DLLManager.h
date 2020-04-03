#pragma once
#ifndef _DLLMANAGER_H_
#define _DLLMANAGER_H_
#include <windows.h>
//负责释放程序所需用到的dll
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