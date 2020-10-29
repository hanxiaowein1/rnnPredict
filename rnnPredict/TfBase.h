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
	std::queue<std::pair<int, tensorflow::Tensor>> tensorQueue;//ÿһ��Ԫ�ض�����batchsize��ͼ��ļ�����
public:
	//���캯��
	TfBase(std::string iniPath, std::string group);
	TfBase(std::string group);
	void construct();
public:
	//��ͨ����
	void Mat2Tensor(std::vector<cv::Mat>& imgs, tensorflow::Tensor& dstTensor);
	//����cv::Mat���õ�Tensor���
	void output(std::vector<cv::Mat>& imgs, std::vector<tensorflow::Tensor>& Output);
	//����Tensor���룬�õ�Tensor���(����tensorInput��batchsize��ͼ��ļ�����)
	void output(tensorflow::Tensor& tensorInput, vector<tensorflow::Tensor>& tensorOutput);

	virtual void convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs);
	virtual bool checkQueueEmpty();
};

void showTensor(tensorflow::Tensor& tensor);

#endif