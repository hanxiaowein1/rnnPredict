#pragma once
#ifndef _TRBASE_H_
#define _TRBASE_H_
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvInfer.h"
#include "NvUffParser.h"
#include <cuda_runtime_api.h>
#include "opencv2/opencv.hpp"
#include "ModelProp.h"
class TrBase : public ModelProp
{
public:
	template <typename T>
	using myUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
public:
	std::queue<std::pair<int, std::vector<float>>> tensorQueue;//ÿһ��Ԫ�ض�����batchsize��ͼ��ļ�����
	//Ϊ�˼򻯴��룬�����ͨ�������Ƴ���ȫ������ModelProp����֤
	//samplesCommon::UffSampleParams mParams;
	myUniquePtr<nvinfer1::IBuilder> mBuilder{ nullptr };
	myUniquePtr<nvinfer1::INetworkDefinition> mNetwork{ nullptr };
	myUniquePtr<nvinfer1::IBuilderConfig> mConfig{ nullptr };
	myUniquePtr<nvuffparser::IUffParser> mParser{ nullptr };
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine{ nullptr };
	samplesCommon::BufferManager* mBuffer{ nullptr };
	myUniquePtr<nvinfer1::IExecutionContext> mContext{ nullptr };
public:
	TrBase(std::string iniPath, std::string group);
	TrBase(std::string group);
	virtual ~TrBase() {};
	virtual bool build(unsigned long long memory, int batchsize);
	virtual bool infer(vector<cv::Mat>& imgs);
	virtual void constructNetwork() = 0;
	virtual bool processInput(vector<cv::Mat>& imgs);
	bool transformInMemory(vector<cv::Mat>& imgs, float* dstPtr);
	virtual unsigned long long getMemory(std::string iniPath, std::string group);
	virtual unsigned long long getMemory(std::string group);

	virtual bool checkQueueEmpty();
	virtual void convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs);
};


#endif