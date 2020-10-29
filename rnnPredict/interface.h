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
RNN_API bool slideProcess(RnnHandle myHandle, const char* slidePath, Anno *annos, int *len, double* wholeScore, UpdateProgressFunc callback);
RNN_API void freeModelMem(RnnHandle myHandle);


#endif
