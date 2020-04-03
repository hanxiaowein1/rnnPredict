#include "SdpcSlideRead.h"



SdpcSlideRead::SdpcSlideRead()
{
}

SdpcSlideRead::SdpcSlideRead(const char* slidePath):SlideRead()
{
	iniHandle(slidePath);
	iniInfo();
	ini_ration();
}

SdpcSlideRead::~SdpcSlideRead()
{
	//关闭sdpc的handle
	closeSdpc(h);
}

void SdpcSlideRead::iniHandle(const char* slidePath)
{
	h = openSdpc((char*)slidePath);
}

void SdpcSlideRead::iniInfo()
{
	::getSdpcInfo(h, &m_sdpcInfo);
}

void SdpcSlideRead::getSlideWidth(int & width)
{
	width = m_sdpcInfo.width;
}

void SdpcSlideRead::getSlideHeight(int & height)
{
	height = m_sdpcInfo.height;
}

void SdpcSlideRead::getSlideBoundX(int & boundX)
{
	boundX = 0;
}

void SdpcSlideRead::getSlideBoundY(int & boundY)
{
	boundY = 0;
}

void SdpcSlideRead::getSlideMpp(double & mpp)
{
	mpp = m_sdpcInfo.mpp;
}

void SdpcSlideRead::getLevelDimensions(int level, int & width, int & height)
{
	width = m_sdpcInfo.w[level];
	height = m_sdpcInfo.h[level];
}

bool SdpcSlideRead::status()
{
	if (h == 0)
		return false;
	return true;
}

void SdpcSlideRead::getTile(int level, int x, int y, int width, int height, cv::Mat &img)
{
	if (width == 0 || height == 0)
		return;
	bufferManage(width, height, 3);
	//从sdpcHandle中读取图像到img中
	::getTile(h, level, y, x, width, height, buffer);//调用全局的函数
	img = cv::Mat(height, width, CV_8UC3, buffer, cv::Mat::AUTO_STEP).clone();
}