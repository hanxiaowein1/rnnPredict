#pragma once
#ifndef _MODEL2HOLDER_H_
#define _MODEL2HOLDER_H_
//#include "model2.h"
#include "TfModel2.h"
#include "TrModel2.h"
#include "MultiImageRead.h"
class Model2Holder
{
public:
	Model2Holder();
	Model2Holder(string iniPath);
	~Model2Holder();
	void runModel2(MultiImageRead& mImgRead, std::vector<regionResult>& rResults);
	void model2Process(std::vector<cv::Mat>& imgs, std::vector<model2Result>& results);
private:
	void enterModel2Queue(MultiImageRead& mImgRead);
	bool popModel2Queue(std::vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	void sortResultsByScore(std::vector<regionResult>& results);
	void model2Config(std::string iniPath);
	void initPara(MultiImageRead &mImgRead);
private:
	//TfModel2* model2Handle = nullptr;
	TrModel2* model2Handle = nullptr;
	int model1Height;
	int model1Width;
	float model1Mpp;
	int model2Height;
	int model2Width;
	float model2Mpp;

	int slideHeight;
	int slideWidth;
	double slideMpp;

	std::atomic<bool> enterFlag2 = false;
	std::mutex queue2Lock;
	queue<std::pair<cv::Rect, cv::Mat>> model2Queue;
	std::condition_variable queue_cv2;
};

#endif