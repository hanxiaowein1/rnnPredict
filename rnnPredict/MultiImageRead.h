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

	std::condition_variable cv_queue_has_elem;//如果有队列有元素的条件变量
	std::condition_variable cv_queue_overflow;//如果队列元素太多的条件变量
	std::mutex data_mutex;
	std::queue<std::pair<cv::Rect, cv::Mat>> data_queue;
	std::atomic<int> maxQueueNum = 30;
	//std::atomic<int> maxPopNum = 

	using Task = std::function<void()>;
	//线程池
	std::vector<std::thread> pool;
	// 任务队列
	std::condition_variable task_cv;
	std::queue<Task> tasks;
	//对task的锁
	std::mutex task_mutex;
	//保存着指向SlideRead的指针
	std::vector<std::unique_ptr<SlideRead>> sReads;
	std::vector<std::condition_variable> cv_sRead;
	std::vector<std::mutex> sRead_mutex;//为每一个SlideRead建立锁
	std::atomic<int> read_level = 0;

	std::atomic<bool> stopped;
	//bool addTaskFlag = true;
	std::atomic<int> idlThrNum = 8;//判断线程有多少闲置;
	std::atomic<int> totalThrNum = 8;//判断总共有多少线程;

	int TaskCount = 0;
	std::atomic<int> totalTaskCount = 0;//总共的task的数量，数量与rects保持一致。
	std::atomic<int> readTaskCount = 0;
	
	//gamma标志，true表示要做gamma变换，false表示不用做gamma变换
	std::atomic<bool> gamma_flag = true;


public:
	MultiImageRead(const char* slidePath);
	~MultiImageRead();
	std::unique_ptr<SlideRead> getSingleReadHandleAndReleaseOthers();
	void popQueueWithoutLock(std::vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	void createThreadPool();
	//i表示取哪一个SlideRead
	void readTask(int i, cv::Rect rect);
	bool popQueue(std::pair<cv::Rect, cv::Mat> &rectMat);
	//如果队列里面有图，那么就全部pop出，否则只pop出一个
	bool popQueue(std::vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	void setRects(std::vector<cv::Rect> rects);
	void setQueueMaxNum(int num) {
		maxQueueNum = num;
	}
	void setReadLevel(int level) {
		read_level = level;
	}
	void m_GammaCorrection(cv::Mat& src, cv::Mat& dst, float fGamma);
	//设置gamma变换的标志
	void setGammaFlag(bool flag)
	{
		gamma_flag = flag;
	}
	//level0下的宽
	void getSlideWidth(int& width);
	//level0下的高
	void getSlideHeight(int& height);
	//level0下的有效区域x轴起始点
	void getSlideBoundX(int& boundX);
	//level0下的有效区域y轴起始点
	void getSlideBoundY(int& boundY);
	//获得mpp
	void getSlideMpp(double& mpp);
	//获取指定level下的宽，高
	void getLevelDimensions(int level, int& width, int& height);
	//为了方便起见，在MultiImageRead里面也设置单张图像读取
	void getTile(int level, int x, int y, int width, int height, cv::Mat& img);

	//获得各个图层的比例
	int get_ratio();
	bool status();
};


#endif