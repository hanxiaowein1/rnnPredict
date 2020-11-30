#pragma once

#ifndef _MULIMAGEREAD_H_
#define _MULIMAGEREAD_H_

#include <queue>
#include <vector>
#include <memory>
#include <utility>
#include "ThreadPool.h"
#include "SlideRead.h"
#include "SlideFactory.h"
#include "opencv2/opencv.hpp"
#include "multi_thread_queue.h"
class MultiImageRead/* : public MultiThreadQueue<std::pair<cv::Rect, cv::Mat>>*/
{
private:
	std::string m_slidePath;

	std::condition_variable cv_queue_has_elem;//����ж�����Ԫ�ص���������
	std::condition_variable cv_queue_overflow;//�������Ԫ��̫�����������
	std::mutex data_mutex;
	std::queue<std::pair<cv::Rect, cv::Mat>> data_queue;
	std::atomic<int> maxQueueNum = 30;
	//std::atomic<int> maxPopNum = 

	using Task = std::function<void()>;
	//�̳߳�
	std::vector<std::thread> pool;
	// �������
	std::condition_variable task_cv;
	std::queue<Task> tasks;
	//��task����
	std::mutex task_mutex;
	//������ָ��SlideRead��ָ��
	std::vector<std::unique_ptr<SlideRead>> sReads;
	std::vector<std::condition_variable> cv_sRead;
	std::vector<std::mutex> sRead_mutex;//Ϊÿһ��SlideRead������
	std::atomic<int> read_level = 0;

	std::atomic<bool> stopped;
	//bool addTaskFlag = true;
	std::atomic<int> idlThrNum = 8;//�ж��߳��ж�������;
	std::atomic<int> totalThrNum = 8;//�ж��ܹ��ж����߳�;

	int TaskCount = 0;
	std::atomic<int> totalTaskCount = 0;//�ܹ���task��������������rects����һ�¡�
	std::atomic<int> readTaskCount = 0;
	
	//gamma��־��true��ʾҪ��gamma�任��false��ʾ������gamma�任
	std::atomic<bool> gamma_flag = true;


public:
	MultiImageRead(const char* slidePath);
	~MultiImageRead();
	std::unique_ptr<SlideRead> getSingleReadHandleAndReleaseOthers();
	void popQueueWithoutLock(std::vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	void createThreadPool();
	//i��ʾȡ��һ��SlideRead
	void readTask(int i, cv::Rect rect);
	bool popQueue(std::pair<cv::Rect, cv::Mat> &rectMat);
	//�������������ͼ����ô��ȫ��pop��������ֻpop��һ��
	bool popQueue(std::vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	void setRects(std::vector<cv::Rect> rects);
	void setQueueMaxNum(int num) {
		maxQueueNum = num;
	}
	void setReadLevel(int level) {
		read_level = level;
	}
	void m_GammaCorrection(cv::Mat& src, cv::Mat& dst, float fGamma);
	//����gamma�任�ı�־
	void setGammaFlag(bool flag)
	{
		gamma_flag = flag;
	}
	//level0�µĿ�
	void getSlideWidth(int& width);
	//level0�µĸ�
	void getSlideHeight(int& height);
	//level0�µ���Ч����x����ʼ��
	void getSlideBoundX(int& boundX);
	//level0�µ���Ч����y����ʼ��
	void getSlideBoundY(int& boundY);
	//���mpp
	void getSlideMpp(double& mpp);
	//��ȡָ��level�µĿ���
	void getLevelDimensions(int level, int& width, int& height);
	//Ϊ�˷����������MultiImageRead����Ҳ���õ���ͼ���ȡ
	void getTile(int level, int x, int y, int width, int height, cv::Mat& img);

	//��ø���ͼ��ı���
	int get_ratio();
	bool status();
};


#endif