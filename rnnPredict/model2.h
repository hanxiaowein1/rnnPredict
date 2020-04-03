#pragma once
#include "model.h"
#include "types.h"

#ifndef _MODEL2_H_
#define _MODEL2_H_
class model2 :public model
{
private:
	const float model1Resolution = 0.586f;
	const float model2Resolution = 0.293f;
	float srcResolution = 0.463f;
	int batchsize = 30;
public:
	model2();
	model2(modelConfig config, char* buffer, int size);
	~model2();
	vector<float> resultOutput(const Tensor &tensor);
	void dataInput(const vector<cv::Point> &points, const cv::Mat &inMat, vector<cv::Mat> &imgs);
	vector<float> model2Process(const vector<cv::Rect> &rects, const vector<model1Result> &results, const cv::Mat &inMat, vector<cv::Mat> &imgs);
	vector<float> model2Process(vector<cv::Mat>& imgs);
	vector<float> model2Process(vector<Tensor>& tensors);
	vector<float> model2ProcessResizeInPb(std::vector<cv::Mat>& imgs);
	void model2Process(vector<cv::Mat>& imgs, vector<tensorflow::Tensor> &tensors);
	void setSrcRes(float srcRes) {
		srcResolution = srcRes;
	}
	float getM2Res() {
		return model2Resolution;
	}
};


#endif
