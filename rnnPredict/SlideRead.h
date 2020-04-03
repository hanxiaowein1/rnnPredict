#pragma once
#ifndef _SLIDEREAD_H_
#define _SLIDEREAD_H_
#include <memory>
#include "opencv2/opencv.hpp"
using namespace std;
//��WholeSlide�Ļ���
class SlideRead
{
public:
	unsigned char* buffer = nullptr;
	int m_width = 0;
	int m_height = 0;
	int m_channel = 0;
	int m_ratio;//����level֮��ı���
public:
	SlideRead();
	virtual ~SlideRead();
	//ͳһ�Ĺ���buffer
	void bufferManage(int width, int height, int channel);
	//�õ�ͼ��ÿ���㼶�ı�����ϵ
	void ini_ration();
	
public:
	virtual bool status() = 0;
	//���ݲ㼶����ߣ��õ�һ��RGB��ͼ��
	virtual void getTile(int level, int x, int y, int width, int height, cv::Mat &img) = 0;
	//level0�µĿ�
	virtual void getSlideWidth(int &width) = 0;
	//level0�µĸ�
	virtual void getSlideHeight(int &height) = 0;
	//level0�µ���Ч����x����ʼ��
	virtual void getSlideBoundX(int &boundX) = 0;
	//level0�µ���Ч����y����ʼ��
	virtual void getSlideBoundY(int &boundY) = 0;
	//���mpp
	virtual void getSlideMpp(double &mpp) = 0;
	//��ȡָ��level�µĿ���
	virtual void getLevelDimensions(int level, int &width, int &height) = 0;

};

#endif