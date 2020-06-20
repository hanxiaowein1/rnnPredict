#pragma once
#ifndef _OPENSLIDEREAD_H_
#define _OPENSLIDEREAD_H_
#include "SlideRead.h"
#include "openslide.h"
class OpenSlideRead : public SlideRead
{
private:
	openslide_t* osr = nullptr;
	int m_boundX = 0;
	int m_boundY = 0;
	//std::unique_ptr<uint8_t[]> uBuffer(new uint8_t[width * height * 4]);
	//uint8_t* buffer = nullptr;
public:
	OpenSlideRead();
	OpenSlideRead(const char* slidePath);
	~OpenSlideRead();
public:
	
	virtual bool status();

	virtual void getTile(int level, int x, int y, int width, int height, cv::Mat &img);
	//初始化openslide的handle
	void iniHandle(const char* slidePath);

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

private:
	int get_os_property(openslide_t *slide, const char* propName);
	double get_os_property_double(openslide_t* slide, const char* propName);
};

#endif