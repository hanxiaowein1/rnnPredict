#pragma once
#ifndef _MODEL1HOLDER_H_
#define _MODLE1HOLDER_H_

#include <vector>
#include <atomic>

#include "tensorflow/core/framework/tensor.h"
#include "opencv2/opencv.hpp"
#include "MultiImageRead.h"
#include "model1.h"
//��������model1��ָ��
//����model1������(���̶߳�ͼ����ô��ͼ�ȵ�)
//����model1�Ľ��
class Model1Holder
{
	//���г�Ա����
public:
	Model1Holder(string model1Path);
	~Model1Holder();
	vector<regionResult> runModel1(MultiImageRead& mImgRead);
	//˽�г�Ա����
private:
	//��batchsizeΪ��ֵ����imgs�ŵ�tensors���棬ÿbatchsize��ͼ��ŵ�һ��tensors����
	void Mats2Tensors(std::vector<cv::Mat>& imgs, std::vector<tensorflow::Tensor>& tensors, int batchsize);
	void Mats2Tensors(std::vector<std::pair<cv::Rect, cv::Mat>>& rectMats, std::vector<std::pair<std::vector<cv::Rect>, tensorflow::Tensor>>& rectsTensors, int batchsize);
	void normalize(std::vector<cv::Mat>& imgs, tensorflow::Tensor& tensor);
	void enterModel1Queue4(std::atomic<bool>& flag, MultiImageRead& mImgRead);
	vector<cv::Rect> iniRects(int sHeight, int sWidth, int height, int width, int overlap, bool flag_right, bool flag_down);
	vector<cv::Rect> get_rects_slide();
	bool popModel1Queue(vector<std::pair<cv::Rect, cv::Mat>> &rectMats/*vector<std::pair<vector<cv::Rect>, Tensor>>& rectsTensors*/);
	bool checkFlags();
	void model1Config(string model1Path);
	bool iniPara(MultiImageRead& mImgRead);
	bool initialize_binImg(MultiImageRead& mImgRead);
	void threshold_segmentation(cv::Mat& img, cv::Mat& binImg, int level, int thre_col, int thre_vol);
	void remove_small_objects(cv::Mat& binImg, int thre_vol);
	//˽�г�Ա����
private:
	model1* model1Handle = nullptr;
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

	std::atomic<bool> enterFlag3 = false;
	std::atomic<bool> enterFlag4 = false;
	std::atomic<bool> enterFlag5 = false;
	std::atomic<bool> enterFlag6 = false;
	std::mutex queue_lock;
	std::condition_variable queue_cv;
	//std::queue<std::pair<vector<cv::Rect>, Tensor>> data_queue;
	std::queue<std::pair<cv::Rect, cv::Mat>> data_queue2;
};

#endif