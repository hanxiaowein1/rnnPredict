#pragma once
#ifndef _MODEL2HOLDER_H_
#define _MODEL2HOLDER_H_
#include "model2.h"
#include "MultiImageRead.h"
class Model2Holder
{
public:
	Model2Holder();
	Model2Holder(string model2Path);
	~Model2Holder();
	void runModel2(MultiImageRead& mImgRead, vector<regionResult>& rResults);
	void model2Process(vector<cv::Mat>& imgs, vector<Tensor>& tensors);
private:
	void enterModel2Queue(MultiImageRead& mImgRead);
	bool popModel2Queue(vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	void sortResultsByScore(vector<regionResult>& results);
	void model2Config(string model2Path);
	void initPara(MultiImageRead &mImgRead);
private:
	model2* model2Handle = nullptr;
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