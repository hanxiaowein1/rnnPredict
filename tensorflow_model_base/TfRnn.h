#pragma once
#ifndef _TFRNN_H_
#define _TFRNN_H_
#include "TfBase.h"
//��Ϊrnn���������ԣ���������һ������������һ��ͼ�������޷�ʹ��ͳһ�Ľӿڣ�ֻ��ʹ���Լ������Ľӿ���
class TfRnn : public TfBase
{
public:
	TfRnn(std::string iniPath, std::string group);
	TfRnn(std::string group);
	~TfRnn();
	//����Ϊmodel2�����
	std::vector<float> rnnProcess(std::vector<std::vector<float>>& input);
	std::vector<float> rnnProcess(tensorflow::Tensor& tensorInput);
private:
	std::vector<float> resultOutput(tensorflow::Tensor& tensor);
	virtual void clearResult() {}
	virtual void convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs) {}
	virtual int processFirstDataInQueue() { return 1; }
	virtual void processInBatch(std::vector<cv::Mat>& imgs) {};
};

#endif