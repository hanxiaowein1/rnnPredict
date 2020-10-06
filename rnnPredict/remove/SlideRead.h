#pragma once
#ifndef _SLIDEREAD_H_
#define _SLIDEREAD_H_
#include <memory>
#include "opencv2/opencv.hpp"
using namespace std;
//读WholeSlide的基类
class SlideRead
{
public:
	unsigned char* buffer = nullptr;
	int m_width = 0;
	int m_height = 0;
	int m_channel = 0;
	int m_ratio;//各个level之间的比例
public:
	SlideRead();
	virtual ~SlideRead();
	//统一的管理buffer
	void bufferManage(int width, int height, int channel);
	//得到图像每个层级的比例关系
	void ini_ration();
	
public:
	virtual bool status() = 0;
	//根据层级，宽高，得到一张RGB的图像
	virtual void getTile(int level, int x, int y, int width, int height, cv::Mat &img) = 0;
	//level0下的宽
	virtual void getSlideWidth(int &width) = 0;
	//level0下的高
	virtual void getSlideHeight(int &height) = 0;
	//level0下的有效区域x轴起始点
	virtual void getSlideBoundX(int &boundX) = 0;
	//level0下的有效区域y轴起始点
	virtual void getSlideBoundY(int &boundY) = 0;
	//获得mpp
	virtual void getSlideMpp(double &mpp) = 0;
	//获取指定level下的宽，高
	virtual void getLevelDimensions(int level, int &width, int &height) = 0;

};

#endif