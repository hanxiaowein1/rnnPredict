#pragma once
#ifndef _MODEL3HOLDER_H_
#define _MODEL3HOLDER_H_
#include "model3.h"
#include "MultiImageRead.h"
class Model3Holder
{
public:
	Model3Holder();
	Model3Holder(string model3Path);
	~Model3Holder();
	vector<PointScore> runModel3(MultiImageRead &mImgRead, vector<Anno>& annos);
private:
	void enterModel2Queue(MultiImageRead& mImgRead);
	bool popModel2Queue(vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	vector<PointScore> model3Recom(vector<std::pair<cv::Rect, model3Result>>& xyResults);
	cv::Point rect2Point(int x, int y, float radius);
	void model3Config(string model3Path);
	void initPara(MultiImageRead& mImgRead);
private:
	model3* model3Handle = nullptr;
	int model3Height;
	int model3Width;
	float model3Mpp;

	int slideHeight;
	int slideWidth;
	double slideMpp;

	std::atomic<bool> enterFlag2 = false;
	std::condition_variable queue_cv2;
	std::mutex queue2Lock;
	queue<std::pair<cv::Rect, cv::Mat>> model2Queue;
};

#endif