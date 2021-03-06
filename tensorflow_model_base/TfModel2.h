#pragma once
#ifndef _TFMODEL2_H_
#define _TFMODEL2_H_
#include "TfBase.h"
#include "types.h"
class TfModel2 : public TfBase
{
public:
	std::vector<model2Result> m_results;
public:
	TfModel2(std::string iniPath, std::string group);
	TfModel2(std::string group);
	virtual void processInBatch(std::vector<cv::Mat> &imgs);
	virtual std::string getGroup() { return "TfModel2"; }
	virtual int processFirstDataInQueue();
	virtual void clearResult() {
		m_results.clear();
	}
private:
	std::vector<float> resultOutput(tensorflow::Tensor& tensor);
	std::vector<model2Result> resultOutput(std::vector<tensorflow::Tensor>& tensors);
};

#endif