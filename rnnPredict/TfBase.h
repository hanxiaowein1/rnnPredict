#pragma once
#ifndef _TFBASE_H_
#define _TFBASE_H_

#ifndef COMPILER_MSVC
#define COMPILER_MSVC
#endif //COMPILER_MSVC

#ifndef NOMINMAX
#define NOMINMAX
#endif //NOMINMAX

#include<iostream>
#include<string>
#include<vector>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "Eigen/Dense"

#include "ModelProp.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>

using namespace std;

class TfBase : public ModelProp
{
private:
	std::unique_ptr<tensorflow::Session> m_session;
public:
	std::queue<tensorflow::Tensor> tensorQueue;//每一个元素都是以batchsize个图像的集合体
public:
	//构造函数
	TfBase(std::string iniPath, std::string group);
	void construct();
public:
	//普通函数
	void Mat2Tensor(std::vector<cv::Mat>& imgs, tensorflow::Tensor& dstTensor);
	//传入cv::Mat，得到Tensor输出
	void output(std::vector<cv::Mat>& imgs, std::vector<tensorflow::Tensor>& Output);
	//传入Tensor输入，得到Tensor输出(其中tensorInput是batchsize个图像的集合体)
	void output(tensorflow::Tensor& tensorInput, vector<tensorflow::Tensor>& tensorOutput);

	virtual void convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs);
	virtual bool checkQueueEmpty();
};

#endif