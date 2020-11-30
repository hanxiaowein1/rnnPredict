#pragma once
#ifndef _MODELPROP_H_
#define _MODELPROP_H_

#include <string>
#include <vector>
#include <functional>
#include <windows.h>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <future>
#include "opencv2/opencv.hpp"
#include "my_thread.h"

class ModelInputProp
{
public:
	int batchsize;
	int height;
	int width;
	int channel;
	double mpp;
public:
	//��ini�ļ��г�ʼ��
	void initByiniFile(std::string iniPath, std::string group);
	void initByIniConfig(std::string group);
};

class ModelFileProp
{
public:
	std::string inputName;
	std::vector<std::string> outputNames;
	std::string filepath;
public:
	void initByiniFile(std::string iniPath, std::string group);
	void initByIniConfig(std::string group);
};

class ModelProp : public MyThread
{
protected:
	ModelInputProp inputProp;
	ModelFileProp fileProp;
public:
	std::mutex queue_lock;
	std::condition_variable tensor_queue_cv;

	//std::function<void(std::vector<cv::Mat>&)> task;//����������洦��һ��batchsize�ĺ���
							   //��������һ�£����Ҫ����task����ô����Ĳ���һ�������ã����ԣ�����Ҫһ����Ա��������result������task�Ĳ�����Ȼ��ÿ���޸������Ա����
							   //���ҵ���taskѭ������Ҫ�������Ա��������ſ��ԣ���ô����Ҫ�ڶ���task���У�������д���麯��clear()����Model1��Model2����������м���ʵ�ּ���
							   //ʵ�־���clear���Լ����еĽ����perfect�����Ҽ�ֱ�Ǹ��˲ţ��ſ�һ���ع��������õ������������ˣ�66666
public:
	virtual ~ModelProp();
	virtual void resizeImages(std::vector<cv::Mat>& imgs, int height, int width);

	//�������������ò�����
	virtual void process(std::vector<cv::Mat>& imgs);
	//��batchsizeΪ��λ����imgs����inFunc����
	virtual void process2(std::vector<cv::Mat>& imgs, std::function<void(std::vector<cv::Mat>&)> inFunc);
	virtual void processInBatch(std::vector<cv::Mat>& imgs) = 0;


	virtual void convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs) = 0;
	virtual void clearResult() = 0;
	virtual void processDataConcurrency(std::vector<cv::Mat>& imgs);
	virtual bool checkQueueEmpty() = 0;
	virtual int processFirstDataInQueue() = 0;
};

#endif