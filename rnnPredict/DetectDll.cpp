#include "DetectDll.h"

//DetectDll::DetectDll() : m_detect_dll(LoadLibraryA("Caffe2Base.dll"), FreeLibrary),
//getDetectHandle((getDetectHandle_export)GetProcAddress(m_detect_dll.get(), "getDetectHandle")),
//runModel((runModel_export)GetProcAddress(m_detect_dll.get(), "runModel")),
//releaseDetectHandle((releaseDetectHandle_export)GetProcAddress(m_detect_dll.get(), "releaseDetectHandle")),
//handle(getDetectHandle(), releaseDetectHandle)
//{
//	
//}

DetectDll::DetectDll(std::string config_path)
{
	m_dll_handle = LoadLibraryA("Caffe2Base.dll");
	getFunctionFromDll();
	m_detect_handle = getDetectHandle(config_path);
}

DetectDll::~DetectDll()
{
	FreeLibrary(m_dll_handle);
}

void DetectDll::getFunctionFromDll()
{
	if (m_dll_handle == nullptr)
	{
		std::cout << "Caffe2Base.dll load failed" << std::endl;
		return;
	}
	getDetectHandle = (getDetectHandle_export)GetProcAddress(m_dll_handle, "getDetectHandle");
	if (getDetectHandle == nullptr)
	{
		std::cout << "getDetectHandle failed" << std::endl;
		return;
	}
	runModel = (runModel_export)GetProcAddress(m_dll_handle, "runModel");
	if (runModel == nullptr)
	{
		std::cout << "runModel failed" << std::endl;
		return;
	}
	releaseDetectHandle = (releaseDetectHandle_export)GetProcAddress(m_dll_handle, "releaseDetectHandle");
	if (releaseDetectHandle == nullptr)
	{
		std::cout << "releaseDetectHandle failed" << std::endl;
		return;
	}
}

DetectResult DetectDll::runModelInterface(std::vector<cv::Mat>& imgs)
{
	auto result = runModel(m_detect_handle, imgs);
	return result;
}
