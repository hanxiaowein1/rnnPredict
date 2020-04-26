#pragma once
#ifndef _MODEL2HOLDER_H_
#define _MODEL2HOLDER_H_
//#include "model2.h"
#include "TfModel2.h"
#include "TrModel2.h"
#include "MultiImageRead.h"
class Model2Holder
{
public:
	Model2Holder();
	Model2Holder(string iniPath);
	~Model2Holder();
	void runModel2(MultiImageRead& mImgRead, std::vector<regionResult>& rResults);
	void model2Process(std::vector<cv::Mat>& imgs, std::vector<model2Result>& results);
	void createThreadPool(int threadNum);
private:
	void popQueueWithoutLock(vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	void pushData(MultiImageRead& mImgRead);
	void enterModel2Queue(MultiImageRead& mImgRead);
	bool popData(std::vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	bool popModel2Queue2(std::vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	void sortResultsByScore(std::vector<regionResult>& results);
	void model2Config(std::string iniPath);
	void initPara(MultiImageRead &mImgRead);
private:
	//TfModel2* model2Handle = nullptr;
	TrModel2* model2Handle = nullptr;
	int model1Height;
	int model1Width;
	float model1Mpp;
	int model2Height;
	int model2Width;
	float model2Mpp;

	int slideHeight;
	int slideWidth;
	double slideMpp;

	std::mutex data_mutex;
	std::condition_variable data_cv;
	std::queue<std::pair<cv::Rect, cv::Mat>> data_queue;

	//���Ķ��̵߳���ʽ�������̳߳أ�Ȼ���ظ���task�Ƶ�tasks����(�����ճ���ǰ����·����)
	using Task = std::function<void()>;
	//thread pool
	std::vector<std::thread> pool;
	// task
	std::condition_variable task_cv;
	std::queue<Task> tasks;
	std::mutex task_mutex;
	std::atomic<bool> stopped;//ֹͣ�̵߳ı�־
	std::atomic<int> idlThrNum = 1;//�����߳�����
	std::atomic<int> totalThrNum = 1;//�ܹ��߳�����
};

#endif