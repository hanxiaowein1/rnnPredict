#pragma once
#ifndef _DETECTEXPORT_H_
#define _DETECTEXPORT_H_
#include <string>
#include <map>
#include <memory>
#include "DetectModel.h"

#ifdef _DETECTMODEL_EXPORT_
#define DETECT_API extern "C" __declspec(dllexport)
#else 
#define DETECT_API extern "C" __declspec(dllimport)
#endif // !TenCExport

typedef unsigned long long* DetectHandle;

DETECT_API DetectHandle getDetectHandle(std::string config_path);
DETECT_API DetectResult runModel(DetectHandle handle, std::vector<cv::Mat>& imgs);
DETECT_API void releaseDetectHandle(DetectHandle handle);


#endif