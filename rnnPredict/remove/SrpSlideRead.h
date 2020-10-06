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
typedef bool(*BeginBatch_function)(SrpCtx ctx);
typedef void (*WriteAnno_function)(SrpCtx ctx, Anno* annos, int count);
typedef bool(*EndBatch_function)(SrpCtx ctx);
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
	BeginBatch_function BeginBatch = nullptr;
	WriteAnno_function WriteAnno = nullptr;
	EndBatch_function EndBatch = nullptr;
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

	//level0�µĿ�
	virtual void getSlideWidth(int& width);
	//level0�µĸ�
	virtual void getSlideHeight(int& height);
	//level0�µ���Ч����x����ʼ��
	virtual void getSlideBoundX(int& boundX);
	//level0�µ���Ч����y����ʼ��
	virtual void getSlideBoundY(int& boundY);
	//���mpp
	virtual void getSlideMpp(double& mpp);
	//��ȡָ��level�µĿ���
	virtual void getLevelDimensions(int level, int& width, int& height);

	void callCleanAnno();
	bool callBeginBatch();
	void callWriteAnno(Anno *anno, int count);
	bool callEndBatch();
	void callWriteParamDouble(const char* key, double value);
	
	OpenRW_function getOpenFunction() { return Open; }
	ReadParamInt32_function getReadParamInt32Function() { return ReadParamInt32; }
	ReadParamDouble_function getReadParamDoubleFunction() { return ReadParamDouble; }
	WriteParamDouble_function getWriteParamDoubleFunction() { return WriteParamDouble; }
	ReadRegionRGB_function getReadRegionRGBFunction() { return ReadRegionRGB; }
	CleanAnno_function getCleanAnnoFunction() { return CleanAnno; }
	BeginBatch_function getBeginBatchFunction() { return BeginBatch; }
	WriteAnno_function getWriteAnnoFunction() { return WriteAnno; }
	EndBatch_function getEndBatchFunction() { return EndBatch; }
	Close_function getCloseFunction() { return Close; }
};


#endif