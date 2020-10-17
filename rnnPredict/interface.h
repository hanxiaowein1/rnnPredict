#pragma once
#ifndef _INTERFACE_H_
#define _INTERFACE_H_

#include "SlideProc.h"
#include "types.h"
//export function
#ifdef _RNNFUNC_EXPORT_
#define RNN_API extern "C" __declspec(dllexport)
#else 
#define RNN_API extern "C" __declspec(dllimport)
#endif // !TenCExport

RNN_API handle initialize_handle(const char* iniPath);
RNN_API bool slideProcess(handle myHandle, const char* slidePath, Anno *annos, int *len, double* wholeScore);
RNN_API void freeModelMem(handle myHandle);


#endif
