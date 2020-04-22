#pragma once
#ifndef _TFRNN_H_
#define _TFRNN_H_
#include "TfBase.h"
//��Ϊrnn���������ԣ���������һ������������һ��ͼ�������޷�ʹ��ͳһ�Ľӿڣ�ֻ��ʹ���Լ������Ľӿ���
class TfRnn : public TfBase
{
public:
	TfRnn(string iniPath, string group);
	~TfRnn();
	//����Ϊmodel2�����
	vector<float> rnnProcess(vector<vector<float>>& input);
	vector<float> rnnProcess(tensorflow::Tensor& tensorInput);	
private:
	vector<float> resultOutput(tensorflow::Tensor& tensor);
	virtual void clearResult() {}
	virtual void convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs) {}
	virtual void processFirstDataInQueue() {}
	virtual void processInBatch(std::vector<cv::Mat>& imgs) {};
};

#endif