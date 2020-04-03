#pragma once
#ifndef _SRPREAD_H_
#define _SRPREAD_H_
#include "SlideRead.h"
#include "types.h"
#include <windows.h>
#include <string>

typedef uint64_t SrpCtx;

typedef SrpCtx(*OpenRW_function)(const char* name);
typedef bool(*ReadParamInt32_function)(SrpCtx ctx, const char* key, int* value);
typedef bool(*ReadParamDouble_function)(SrpCtx ctx, const char* key, double* value);
typedef bool(*WriteParamDouble_function)(SrpCtx ctx, const char* key, double value);
typedef bool(*ReadRegionRGB_function)(SrpCtx ctx, int level, int px, int py, int w, int h, unsigned char* buf, int* plen);
typedef void(*CleanAnno_function)(SrpCtx ctx);
typedef void (*WriteAnno_function)(SrpCtx ctx, Anno* annos, int count);
typedef void(*Close_function)(SrpCtx ctx);

class SrpSlideRead : public SlideRead
{
private:
	SrpCtx m_srpHandle = 0;
	HINSTANCE srpDll = nullptr;
	OpenRW_function Open = nullptr;
	ReadParamInt32_function ReadParamInt32 = nullptr;
	ReadParamDouble_function ReadParamDouble = nullptr;
	WriteParamDouble_function WriteParamDouble = nullptr;
	ReadRegionRGB_function ReadRegionRGB = nullptr;
	CleanAnno_function CleanAnno = nullptr;
	WriteAnno_function WriteAnno = nullptr;
	Close_function Close = nullptr;
public:
	SrpSlideRead();
	SrpSlideRead(const char* slidePath);
	~SrpSlideRead();
public:
	void iniFunction();
	void iniHandle(const char* slidePath);
	
	virtual bool status();
	virtual void getTile(int level, int x, int y, int width, int height, cv::Mat& img);

	//level0下的宽
	virtual void getSlideWidth(int& width);
	//level0下的高
	virtual void getSlideHeight(int& height);
	//level0下的有效区域x轴起始点
	virtual void getSlideBoundX(int& boundX);
	//level0下的有效区域y轴起始点
	virtual void getSlideBoundY(int& boundY);
	//获得mpp
	virtual void getSlideMpp(double& mpp);
	//获取指定level下的宽，高
	virtual void getLevelDimensions(int level, int& width, int& height);

	void callCleanAnno();
	void callWriteAnno(Anno *anno, int count);
	void callWriteParamDouble(const char* key, double value);
	
	OpenRW_function getOpenFunction() { return Open; }
	ReadParamInt32_function getReadParamInt32Function() { return ReadParamInt32; }
	ReadParamDouble_function getReadParamDoubleFunction() { return ReadParamDouble; }
	WriteParamDouble_function getWriteParamDoubleFunction() { return WriteParamDouble; }
	ReadRegionRGB_function getReadRegionRGBFunction() { return ReadRegionRGB; }
	CleanAnno_function getCleanAnnoFunction() { return CleanAnno; }
	WriteAnno_function getWriteAnnoFunction() { return WriteAnno; }
	Close_function getCloseFunction() { return Close; }
};


#endif