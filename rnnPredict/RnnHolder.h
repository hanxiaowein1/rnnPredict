#pragma once
#ifndef _RNNHOLDER_H_
#define _RNNHOLDER_H_

//#include "rnn.h"
#include "TfRnn.h"
#include "MultiImageRead.h"
class RnnHolder
{
public:
	RnnHolder();
	RnnHolder(string rnnParentPath);
	~RnnHolder();
	//输入为model2的十张图像的输出
	float runRnn(tensorflow::Tensor& tensor);
	float runRnn(std::vector<model2Result>& results);
private:
	void rnnConfig(string iniPath);
	float runRnnThread2(int i, tensorflow::Tensor& inputTensor);
	float outputSix(vector<float>& rnnResults_f);
private:
	//vector<rnn*> rnnHandle;
	vector<TfRnn*> rnnHandle;

};

#endif