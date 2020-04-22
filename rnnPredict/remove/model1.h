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
	//����һ����Ұ��
	vector<model1Result> model1Process(const cv::Mat &inMat);
	//�Զ��С�����Ԥ��
	vector<model1Result> model1Process(std::vector<cv::Mat>& imgs);
	vector<model1Result> model1ProcessResizeInPb(std::vector<cv::Mat>& imgs);
	//ֱ���ù�һ����tensor�õ����ս��
	vector<model1Result> model1Process(std::vector<Tensor>& tensorInput);
	//���outputTensor�õ����յĽ��
	vector<model1Result> resultOutput(vector<tensorflow::Tensor> &tensors);
	void TensorToMat(Tensor mask, Mat *dst);
	vector<cv::Point> getRegionPoints2(Mat *mask, float threshold);

	/*
	height:Ҫ�ü���ͼ��ĸ�
	width:Ҫ�ü���ͼ��Ŀ�
	sHeight:Сͼ�ĸ�
	sWidth:Сͼ�Ŀ�
	*/
	void iniRects(int height, int width, int sHeight, int sWidth);

	vector<cv::Rect> getRects() { return m_rects; }

	//��model1�ֱ����µ�����תΪԭʼ�ֱ����µ�����
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
