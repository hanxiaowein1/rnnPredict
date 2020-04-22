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
#include "types.h"

class ModelInputProp
{
public:
	int height;
	int width;
	int channel;
	int batchsize;
	double mpp;
public:
	//从ini文件中初始化
	void initByiniFile(std::string iniPath, std::string group);
};

class ModelFileProp
{
public:
	std::string inputName;
	std::vector<std::string> outputNames;
	std::string filepath;
public:
	void initByiniFile(std::string iniPath, std::string group);
};

class ModelProp
{
public:
	ModelInputProp inputProp;
	ModelFileProp fileProp;
	std::mutex queue_lock;
	std::condition_variable tensor_queue_cv;

	using Task = std::function<void()>;
	//线程池
	std::vector<std::thread> pool;//开启一个线程
	// 任务队列
	std::condition_variable cv_task;
	std::queue<Task> tasks;
	//对task的锁
	std::mutex m_lock;
	std::atomic<bool> stopped;//停止线程的标志
	std::atomic<int> idlThrNum = 1;//判断线程有多少闲置;
	std::atomic<int> totalThrNum = 1;//判断总共有多少线程;

	//std::function<void(std::vector<cv::Mat>&)> task;//这个用来保存处理一个batchsize的函数
							   //而且想了一下，如果要调用task，那么传入的参数一定是引用，所以，还需要一个成员变量（即result）当做task的参数，然后每次修改这个成员变量
							   //而且调用task循环外需要将这个成员变量归零才可以，那么就需要第二个task才行，或者我写个虚函数clear()，让Model1和Model2这两个类进行集成实现即可
							   //实现就是clear掉自己类中的结果，perfect啊！我简直是个人才，才看一点重构，就能用到生产环境中了，66666
public:
	virtual ~ModelProp();
	virtual void resizeImages(std::vector<cv::Mat>& imgs, int height, int width);
	virtual void process(std::vector<cv::Mat>& imgs);
	//以batchsize为单位，对imgs进行inFunc处理
	virtual void process2(std::vector<cv::Mat>& imgs, std::function<void(std::vector<cv::Mat>&)> inFunc);
	virtual void processInBatch(std::vector<cv::Mat>& imgs) = 0;
	virtual void convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs) = 0;
	virtual void clearResult() = 0;
	virtual void createThreadPool();
	virtual void processDataConcurrency(std::vector<cv::Mat>& imgs);
	virtual bool checkQueueEmpty() = 0;
	virtual void processFirstDataInQueue() = 0;
};

#endif