#pragma once

#ifndef _AILAB_DETECTMODEL_H_
#define _AILAB_DETECTMODEL_H_

#include "Caffe2Base.h"
class DetectModel : public Caffe2Base
{
public:
	//std::vector<std::map<std::string, std::string>> m_results;
	std::map<std::string, std::string> m_results;
public:
	DetectModel(std::string group);
	
public:
	void processFirstDataInQueue();
	virtual void clearResult();
};

#endif