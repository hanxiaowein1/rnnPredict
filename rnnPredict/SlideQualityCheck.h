#pragma once
#ifndef _SLIDEQUALITYCHECK_H_
#define _SLIDEQUALITYCHECK_H_
#include "MultiImageRead.h"

//用来监测片子的质量
class SlideQualityCheck
{
private:
	void process2(std::string slidePath, std::string origin);
	float myLaplacian(cv::Mat& img, cv::Mat& binImg, int thre_col = 20);
	//double computeLaplacian(cv::Mat& img, nucSegResult& result, int count, int& lap_nums);

public:
	SlideQualityCheck();
	//为了快速起见，设置mImgRead多线程读取图像
	SlideQualityCheck(MultiImageRead& mImgRead);
	~SlideQualityCheck();
};

#endif

