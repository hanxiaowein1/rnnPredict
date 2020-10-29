#pragma once
#ifndef _TRMODEL2_H_
#define _TRMODEL2_H_
#include "TrBase.h"
#include "types.h"
class TrModel2 : public TrBase
{
public:
	std::vector<model2Result> m_results;
public:
	TrModel2(std::string iniPath, std::string group);
	TrModel2(std::string group);
	virtual std::string getGroup() { return "TrModel2"; }
	virtual int processFirstDataInQueue();
	virtual void clearResult() {
		m_results.clear();
	}
private:
	virtual void constructNetwork();
	bool processOutput(int size, vector<float>& scores);
	bool processOutput2(int size, std::vector<std::vector<float>>& tensors);
	vector<model2Result> resultOutput(int size);
	virtual void processInBatch(std::vector<cv::Mat>& imgs);

	
};

#endif