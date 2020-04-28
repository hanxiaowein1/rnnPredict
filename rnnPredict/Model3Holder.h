#pragma once
#ifndef _MODEL3HOLDER_H_
#define _MODEL3HOLDER_H_
//#include "model3.h"
#include "TfModel3.h"
#include "MultiImageRead.h"
class Model3Holder
{
public:
	Model3Holder();
	Model3Holder(string iniPath);
	~Model3Holder();
	vector<PointScore> runModel3(MultiImageRead &mImgRead, vector<Anno>& annos);
	void createThreadPool(int threadNum);
private:
	//void enterModel2Queue(MultiImageRead& mImgRead);
	void pushData(MultiImageRead& mImgRead);
	void popQueueWithoutLock(vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	bool popData(std::vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	//bool popModel2Queue(vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	vector<PointScore> model3Recom(vector<std::pair<cv::Rect, model3Result>>& xyResults);
	cv::Point rect2Point(int x, int y, float radius);
	void model3Config(string iniPath);
	void initPara(MultiImageRead& mImgRead);
private:
	TfModel3* model3Handle = nullptr;
	int model3Height;
	int model3Width;
	float model3Mpp;

	int slideHeight;
	int slideWidth;
	double slideMpp;

	//std::atomic<bool> enterFlag2 = false;
	std::condition_variable data_cv;
	std::mutex data_mutex;
	queue<std::pair<cv::Rect, cv::Mat>> data_queue;

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