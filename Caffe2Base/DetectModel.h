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
	//��������Ժ��ò����ˣ���processFirstDataInQueue������.
	virtual void processInBatch(std::vector<cv::Mat>& imgs) {
	}
};

#endif