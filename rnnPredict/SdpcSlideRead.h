#pragma once
#ifndef _SDPCREAD_H_
#define _SDPCREAD_H_
#include "SlideRead.h"
#include "sdpcsdk.h"
class SdpcSlideRead : public SlideRead
{
private:
	//��sdpc�ļ���handle
	SdpcHandler h = 0;
	SdpcInfo m_sdpcInfo;
public:
	SdpcSlideRead();
	SdpcSlideRead(const char* slidePath);
	~SdpcSlideRead();
public:
	virtual bool status();
	virtual void getTile(int level, int x, int y, int width, int height, cv::Mat &img);
	//��ʼ��sdpc��handle
	void iniHandle(const char* slidePath);
	//���sdpcƬ�ӵ���Ϣ
	void iniInfo();
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

	SdpcInfo getSdpcInfo() {
		return m_sdpcInfo;
	}
};

#endif