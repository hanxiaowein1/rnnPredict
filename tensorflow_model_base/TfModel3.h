#pragma once
#ifndef _TFMODEL3_H_
#define _TFMODEL3_H_

#include "TfBase.h"
#include "types.h"
class TfModel3 : public TfBase
{
public:
	std::vector<model3Result> m_results;
public:
	TfModel3(std::string iniPath, std::string group);
	TfModel3(std::string group);
	virtual void processInBatch(std::vector<cv::Mat>& imgs);
	virtual std::string getGroup() { return "TfModel3"; }
	virtual int processFirstDataInQueue();
	virtual void clearResult() {
		m_results.clear();
	}
private:
	std::vector<model3Result> resultOutput(const tensorflow::Tensor& tensor);

};

#endif