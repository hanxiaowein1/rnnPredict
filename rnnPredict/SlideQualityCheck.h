#pragma once
#ifndef _SLIDEQUALITYCHECK_H_
#define _SLIDEQUALITYCHECK_H_
#include "MultiImageRead.h"

//�������Ƭ�ӵ�����
class SlideQualityCheck
{
private:
	void process2(string slidePath, string origin);
	float myLaplacian(cv::Mat& img, cv::Mat& binImg, int thre_col = 20);
	//double computeLaplacian(cv::Mat& img, nucSegResult& result, int count, int& lap_nums);

public:
	SlideQualityCheck();
	//Ϊ�˿������������mImgRead���̶߳�ȡͼ��
	SlideQualityCheck(MultiImageRead& mImgRead);
	~SlideQualityCheck();
};

#endif

