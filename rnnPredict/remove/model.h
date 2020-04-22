#pragma once
#ifndef _MODEL_H_
#define _MODEL_H_

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

using namespace tensorflow;
using namespace cv;
using namespace std;
struct modelConfig
{
	int width;
	int height;
	int channel;
	std::string opsInput;
	std::vector<std::string> opsOutput;
};

class model
{
private:
	//设置模型的输入输出
	int m_width;
	int m_height;
	int m_channel;
	std::string m_opsInput;
	std::vector<std::string> m_opsOutput;
	std::unique_ptr<Session> m_session;
public:
	//构造函数
	model();
	model(modelConfig config, char* buffer, int size);
public:
	//普通函数
	//通过传入cv::Mat图像，得到输出
	void output(std::vector<cv::Mat> &imgs, std::vector<Tensor> &Output);
	void output(std::vector<cv::Mat>& imgs, std::vector<Tensor>& Output, int flag);
	//带有batchsize的运算
	void output(Tensor &tensorInput, vector<Tensor> &tensorOutput);
	int getModelWidth() { return m_width; }
	int getModelHeight() { return m_height; }
};




#endif
