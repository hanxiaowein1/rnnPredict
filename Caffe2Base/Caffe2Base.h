#pragma once

#ifndef _AILAB_CAFFE2BASE_H_
#define _AILAB_CAFFE2BASE_H_

#include "c10/util/Flags.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/common.h"
#include "caffe2/core/init.h"
#include "caffe2/core/net.h"
#include "caffe2/core/workspace.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/context_gpu.h"

#include "ModelProp.h"

class Caffe2Base : public ModelProp
{
public:
	caffe2::Workspace work_space;
	caffe2::NetDef initNet_, predictNet_;
public:
	std::queue<std::pair<int, std::vector<float>>> tensorQueue;//现在改个需求，一个元素我还是需要知道原来是有多少张图像

public:
	void construct();
	virtual void convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs);
	virtual bool checkQueueEmpty();
	virtual void runNet(std::vector<float>& data);

public:
	Caffe2Base(std::string group);
};

#endif