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
#include "IniConfig.h"
#include "multi_thread_queue.h"
//用来保存model1的指针
//处理model1的输入(多线程读图，怎么裁图等等)
//返回model1的结果
class Model1Holder : public MultiThreadQueue<std::pair<cv::Rect, cv::Mat>>
{
public:
	Model1Holder(string model1Path);
	~Model1Holder();
	vector<regionResult> runModel1(MultiImageRead& mImgRead);
	//void createThreadPool(int threadNum);
private:
	void pushData(MultiImageRead& mImgRead);
	vector<cv::Rect> iniRects(int sHeight, int sWidth, int height, int width, int overlap, bool flag_right, bool flag_down, int &rows, int &cols);
	vector<cv::Rect> get_rects_slide();
	void model1Config(string iniPath);
	bool iniPara(MultiImageRead& mImgRead);
	bool initialize_binImg(MultiImageRead& mImgRead);
	void threshold_segmentation(cv::Mat& img, cv::Mat& binImg, int level, int thre_col, int thre_vol);
	void remove_small_objects(cv::Mat& binImg, int thre_vol);
private:
	//std::unique_ptr<handle, Deleter> hhh = nullptr;
	std::pair<std::unique_ptr<TfModel1>, std::unique_ptr<TrModel1>> model1Handle;
	bool use_tr = false;
	//std::vector<std::unique_ptr<>>
	int model1Height;
	int model1Width;
	float model1Mpp;
	int model1_batchsize = 20;
	float model1OverlapRatio = 0.25f;
	int slideHeight;
	int slideWidth;
	double slideMpp;
	//裁取的宽高信息
	int block_height = 8192;
	int block_width = 8192;//在第0图层读取的图像的大小
	int read_level = 1;//model1读取的层级
	int levelBin = 4;
	double slideRatio;
	int m_crop_sum;
	cv::Mat binImg;
	int m_thre_col = 20;//rgb的阈值(与mpp无关)
	int m_thre_vol = 150;//面积的阈值(前景分割)

};

#endif