#pragma once
#ifndef _TRMODEL1_H_
#define _TRMODEL1_H_
#include <queue>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <future>
#include "TrBase.h"
class TrModel1 : public TrBase
{
public:
	std::vector<model1Result> m_results;
public:
	TrModel1(std::string iniPath, std::string group);
	virtual std::string getGroup() { return "TrModel1"; }
	virtual void processFirstDataInQueue();
	virtual void clearResult() {
		m_results.clear();
	};
private:
	virtual void constructNetwork();
	vector<model1Result> resultOutput(int size);
	bool processOutput(int size, vector<std::pair<float, cv::Mat>>& pairElems);
	virtual void processInBatch(std::vector<cv::Mat>& imgs);	
};

#endif