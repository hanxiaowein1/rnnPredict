#pragma once
#ifndef _TFMODEL1_H_
#define _TFMODEL1_H_
#include <queue>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <future>
#include "TfBase.h"
#include "types.h"
//�ҿ��Խ������д�ɶ��̴߳����డ��
//��Ϊ���ⲿ��cv::MatתΪtensorflow::TensorҲ��ʱ�䣬�����߼�Ҳ�ܻ���
//����ֱ�Ӵ���vector<cv::Mat>��Ȼ��������ڲ�����һ��queue<tensorflow::Tensor>��
//���ö��߳̽�cv::MatתΪtensorflow::Tensor
//��ô�ⲿ�߼����ܵ����ڲ����ˣ��������޸ġ�
class TfModel1 : public TfBase
{
public:
	std::vector<model1Result> m_results;
private:
	std::vector<model1Result> resultOutput(std::vector<tensorflow::Tensor>& tensors);
	void TensorToMat(tensorflow::Tensor mask, cv::Mat* dst);

public:
	TfModel1(std::string iniPath, std::string group);
	TfModel1(std::string group);
	~TfModel1();
	virtual void processInBatch(std::vector<cv::Mat>& imgs);
	virtual std::string getGroup() { return "TfModel1"; }

	virtual int processFirstDataInQueue();
	virtual void clearResult() {
		m_results.clear();
	}
};

#endif