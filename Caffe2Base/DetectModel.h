#pragma once

#ifndef _AILAB_DETECTMODEL_H_
#define _AILAB_DETECTMODEL_H_

#include "Caffe2Base.h"

typedef std::map<std::string, std::string> DetectResult;

class DetectModel : public Caffe2Base
{
public:
	DetectResult m_result;
public:
	DetectModel(std::string group);
	
public:
	void processFirstDataInQueue();
	virtual void clearResult();
	//这个函数以后用不着了，由processFirstDataInQueue代替了.
	virtual void processInBatch(std::vector<cv::Mat>& imgs) {
	}
};

#endif