#pragma once
#ifndef _RNNHOLDER_H_
#define _RNNHOLDER_H_

//#include "rnn.h"
#include "TfRnn.h"
//#include "MultiImageRead.h"
#include "types.h"
class RnnHolder
{
public:
	RnnHolder();
	RnnHolder(std::string rnnParentPath);
	~RnnHolder();
	//����Ϊmodel2��ʮ��ͼ������
	float runRnn(tensorflow::Tensor& tensor);
	float runRnn(std::vector<model2Result>& results);
private:
	void rnnConfig(std::string iniPath);
	float runRnnThread2(int i, tensorflow::Tensor& inputTensor);
	float outputSix(std::vector<float>& rnnResults_f);
	float outputSix2(std::vector<float>& rnnResults_f);
private:
	//vector<rnn*> rnnHandle;
	//vector<TfRnn*> rnnHandle;
	std::vector<std::unique_ptr<TfRnn>> rnnHandle;
};

#endif