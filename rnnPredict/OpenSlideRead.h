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
	//��ʼ��openslide��handle
	void iniHandle(const char* slidePath);

	//level0�µĿ�
	virtual void getSlideWidth(int &width);
	//level0�µĸ�
	virtual void getSlideHeight(int &height);
	//level0�µ���Ч����x����ʼ��
	virtual void getSlideBoundX(int &boundX);
	//level0�µ���Ч����y����ʼ��
	virtual void getSlideBoundY(int &boundY);
	//���mpp
	virtual void getSlideMpp(double &mpp);
	//��ȡָ��level�µĿ���
	virtual void getLevelDimensions(int level, int &width, int &height);

private:
	int get_os_property(openslide_t *slide, const char* propName);
	double get_os_property_double(openslide_t* slide, const char* propName);
};

#endif