#pragma once
#ifndef _MODEL2HOLDER_H_
#define _MODEL2HOLDER_H_
//#include "model2.h"
#include "TfModel2.h"
#include "TrModel2.h"
#include "MultiImageRead.h"
#include "multi_thread_queue.h"

class Model2Holder : public MultiThreadQueue<std::pair<cv::Rect, cv::Mat>>
{
public:
	Model2Holder();
	Model2Holder(string iniPath);
	~Model2Holder();
	void runModel2(MultiImageRead& mImgRead, std::vector<regionResult>& rResults);
	void model2Process(std::vector<cv::Mat>& imgs, std::vector<model2Result>& results);
	//void createThreadPool(int threadNum);
	void readImageInOrder(std::vector<cv::Rect> rects, MultiImageRead& mImgRead, std::vector<cv::Mat>& imgs);
private:
	//void popQueueWithoutLock(vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	void pushData(MultiImageRead& mImgRead);
	//bool popData(std::vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	void sortResultsByScore(std::vector<regionResult>& results);
	void model2Config(std::string iniPath);
	void initPara(MultiImageRead &mImgRead);
	void startRead(std::vector<cv::Rect> rects, MultiImageRead& mImgRead);
	
private:
	//std::unique_ptr<TrModel2> model2Handle;
	//std::unique_ptr<TfModel2> model2Handle;
	std::pair<std::unique_ptr<TfModel2>, std::unique_ptr<TrModel2>> model2Handle;
	bool use_tr = false;
	int model1Height;
	int model1Width;
	float model1Mpp;
	int model2Height;
	int model2Width;
	float model2Mpp;

	int slideHeight;
	int slideWidth;
	double slideMpp;
};

#endif