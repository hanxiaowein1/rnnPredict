#pragma once
#ifndef _SDPCREAD_H_
#define _SDPCREAD_H_
#include "SlideRead.h"
#include "sdpcsdk.h"
class SdpcSlideRead : public SlideRead
{
private:
	//读sdpc文件的handle
	SdpcHandler h = 0;
	SdpcInfo m_sdpcInfo;
public:
	SdpcSlideRead();
	SdpcSlideRead(const char* slidePath);
	~SdpcSlideRead();
public:
	virtual bool status();
	virtual void getTile(int level, int x, int y, int width, int height, cv::Mat &img);
	//初始化sdpc的handle
	void iniHandle(const char* slidePath);
	//获得sdpc片子的信息
	void iniInfo();
	//level0下的宽
	virtual void getSlideWidth(int &width);
	//level0下的高
	virtual void getSlideHeight(int &height);
	//level0下的有效区域x轴起始点
	virtual void getSlideBoundX(int &boundX);
	//level0下的有效区域y轴起始点
	virtual void getSlideBoundY(int &boundY);
	//获得mpp
	virtual void getSlideMpp(double &mpp);
	//获取指定level下的宽，高
	virtual void getLevelDimensions(int level, int &width, int &height);

	SdpcInfo getSdpcInfo() {
		return m_sdpcInfo;
	}
};

#endif