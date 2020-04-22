#pragma once

#include "model.h"
#include "types.h"
#ifndef _MODEL1_H_
#define _MODEL1_H_


class model1 :public model
{
private:
	vector<cv::Rect> m_rects;
	const float model1Resolution = 0.586f;
	float srcResolution = 0.463f;
	int batchsize = 20;
public:
	model1();
	model1(modelConfig config, char* buffer, int size);
	~model1();
	//传入一个视野块
	vector<model1Result> model1Process(const cv::Mat &inMat);
	//对多个小块进行预测
	vector<model1Result> model1Process(std::vector<cv::Mat>& imgs);
	vector<model1Result> model1ProcessResizeInPb(std::vector<cv::Mat>& imgs);
	//直接用归一化的tensor得到最终结果
	vector<model1Result> model1Process(std::vector<Tensor>& tensorInput);
	//针对outputTensor得到最终的结果
	vector<model1Result> resultOutput(vector<tensorflow::Tensor> &tensors);
	void TensorToMat(Tensor mask, Mat *dst);
	vector<cv::Point> getRegionPoints2(Mat *mask, float threshold);

	/*
	height:要裁剪的图像的高
	width:要裁剪的图像的宽
	sHeight:小图的高
	sWidth:小图的宽
	*/
	void iniRects(int height, int width, int sHeight, int sWidth);

	vector<cv::Rect> getRects() { return m_rects; }

	//从model1分辨率下的坐标转为原始分辨率下的坐标
	void convertPointM1ToSrc(Point &point);

	void setSrcRes(float srcRes) {
		srcResolution = srcRes;
	}
	void setBatchsize(int bs) {
		batchsize = bs;
	}
	float getM1Resolution() {
		return model1Resolution;
	}
};


#endif // !_MODEL1_H_
