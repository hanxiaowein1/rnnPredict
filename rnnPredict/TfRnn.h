#pragma once
#ifndef _TFRNN_H_
#define _TFRNN_H_
#include "TfBase.h"
//因为rnn具有特殊性，其输入是一个向量而不是一张图像，所以无法使用统一的接口，只能使用自己独立的接口了
class TfRnn : public TfBase
{
public:
	TfRnn(string iniPath, string group);
	~TfRnn();
	//输入为model2的输出
	vector<float> rnnProcess(vector<vector<float>>& input);
	vector<float> rnnProcess(tensorflow::Tensor& tensorInput);	
private:
	vector<float> resultOutput(tensorflow::Tensor& tensor);
	virtual void clearResult() {}
	virtual void convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs) {}
	virtual void processFirstDataInQueue() {}
	virtual void processInBatch(std::vector<cv::Mat>& imgs) {};
};

#endif