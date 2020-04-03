#include "DLLManager.h"

#include <iostream>
DLLManager::DLLManager()
{
	//根据dll的特性，反复LoadLibrary()只会创建一个dll的实例，所以可以由DLLManger管理自己的dll，只负责释放即可(生命周期为整个程序的起始)
	srpDll = LoadLibraryA("srp.dll");
	if (srpDll == nullptr)
	{
		std::cout << "srp dll load failed\n" << std::endl;
	}
	//xgDll = LoadLibraryA("xgdll.dll");
	//if (xgDll == nullptr)
	//{
	//	std::cout << "xgboost dll load failed\n" << std::endl;
	//}
}

DLLManager::~DLLManager()
{
	if (srpDll != nullptr)
	{
		FreeLibrary(srpDll);
	}
	//if (xgDll != nullptr)
	//{
	//	FreeLibrary(xgDll);
	//}

}

bool DLLManager::getState()
{
	if (srpDll != nullptr/* && xgDll != nullptr*/)
	{
		return true;
	}
	else
	{
		return false;
	}
}


