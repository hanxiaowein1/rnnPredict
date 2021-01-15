#pragma once
#ifndef _INTERFACE_H_
#define _INTERFACE_H_
//export function

#include "anno.h"
#include "progress_record.h"

typedef unsigned long long* RnnHandle;

#ifdef _RNNFUNC_EXPORT_
#define RNN_API extern "C" __declspec(dllexport)
#else 
#define RNN_API extern "C" __declspec(dllimport)
#endif // !TenCExport

RNN_API RnnHandle initialize_handle(const char* iniPath);
RNN_API bool slideProcess(
	RnnHandle myHandle, 
	const char* slidePath, Anno *annos, int *len, double* wholeScore, UpdateProgressFunc callback);
RNN_API void freeModelMem(RnnHandle myHandle);


typedef void* RnnJavaHandle;

RNN_API RnnJavaHandle initializeHandleJava(const char* ini_path);
RNN_API bool slideProcessJava(
	RnnJavaHandle handle, 
	const char* slide_path, 
	Anno* annos, int* len, double* wholeScore, UpdateProgressFunc callback, const char* savepath);
RNN_API void freeModelMemJava(RnnJavaHandle handle);
RNN_API void setCudaVisibleDevices(const char* num);
RNN_API void setAdditionalPath(const char* path);
#endif
