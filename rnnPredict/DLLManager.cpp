#include "DLLManager.h"

#include <iostream>
DLLManager::DLLManager()
{
	//����dll�����ԣ�����LoadLibrary()ֻ�ᴴ��һ��dll��ʵ�������Կ�����DLLManger�����Լ���dll��ֻ�����ͷż���(��������Ϊ�����������ʼ)
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


