#pragma once

#ifndef _AILAB_DETECTDLL_H_
#define _AILAB_DETECTDLL_H_
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <memory>
#include <windows.h>
#include "opencv2/opencv.hpp"

typedef unsigned long long* DetectHandle;
typedef std::map<std::string, std::string> DetectResult;

typedef DetectHandle(*getDetectHandle_export)(std::string);
typedef DetectResult(*runModel_export)(DetectHandle handle, std::vector<cv::Mat>& imgs);
typedef void(*releaseDetectHandle_export)(DetectHandle handle);

class DetectDll 
{
public:
	getDetectHandle_export getDetectHandle = nullptr;
	runModel_export runModel = nullptr;
	releaseDetectHandle_export releaseDetectHandle = nullptr;
private:
	class DetectModelDeleter
	{
	public:
		void operator()(DetectHandle handle, std::function<void(DetectHandle)> release_function)
		{
			release_function(handle);
		}
	};
	class LibraryDeleter
	{
	public:
		void operator()(HINSTANCE dll)
		{
			FreeLibrary(dll);
		}
	};
private:
	//std::unique_ptr<std::remove_pointer_t<HMODULE>, BOOL(*)(HMODULE)> m_detect_dll;
	//std::unique_ptr<std::remove_pointer_t<DetectHandle>, releaseDetectHandle_export> handle;
	HINSTANCE m_dll_handle = nullptr;
	DetectHandle m_detect_handle = nullptr;

public:
	DetectDll(std::string config_path);
	~DetectDll();
	void getFunctionFromDll();
	DetectResult runModelInterface(std::vector<cv::Mat> &imgs);
};

#endif