#pragma once
#ifndef _MODEL1HOLDER_H_
#define _MODLE1HOLDER_H_

#include <vector>
#include <atomic>

#include "tensorflow/core/framework/tensor.h"
#include "opencv2/opencv.hpp"
#include "MultiImageRead.h"
#include "TfModel1.h"
#include "TrModel1.h"
//��������model1��ָ��
//����model1������(���̶߳�ͼ����ô��ͼ�ȵ�)
//����model1�Ľ��
class Model1Holder
{
public:
	Model1Holder(string model1Path);
	~Model1Holder();
	vector<regionResult> runModel1(MultiImageRead& mImgRead);
	void createThreadPool(int threadNum);
private:
	void pushData(MultiImageRead& mImgRead);
	vector<cv::Rect> iniRects(int sHeight, int sWidth, int height, int width, int overlap, bool flag_right, bool flag_down);
	vector<cv::Rect> get_rects_slide();
	bool popData(vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	void popQueueWithoutLock(vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	void model1Config(string iniPath);
	bool iniPara(MultiImageRead& mImgRead);
	bool initialize_binImg(MultiImageRead& mImgRead);
	void threshold_segmentation(cv::Mat& img, cv::Mat& binImg, int level, int thre_col, int thre_vol);
	void remove_small_objects(cv::Mat& binImg, int thre_vol);
private:
	//std::unique_ptr<TrModel1> model1Handle;
	std::unique_ptr<TfModel1> model1Handle;
	int model1Height;
	int model1Width;
	float model1Mpp;
	int model1_batchsize = 20;
	float model1OverlapRatio = 0.25f;
	int slideHeight;
	int slideWidth;
	double slideMpp;
	//��ȡ�Ŀ����Ϣ
	int block_height = 8192;
	int block_width = 8192;//�ڵ�0ͼ���ȡ��ͼ��Ĵ�С
	int read_level = 1;//model1��ȡ�Ĳ㼶
	int levelBin = 4;
	double slideRatio;
	int m_crop_sum;
	cv::Mat binImg;
	int m_thre_col = 20;//rgb����ֵ(��mpp�޹�)
	int m_thre_vol = 150;//�������ֵ(ǰ���ָ�)

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

	std::once_flag create_thread_flag;
};

#endif