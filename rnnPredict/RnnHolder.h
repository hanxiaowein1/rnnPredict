#pragma once
#ifndef _RNNHOLDER_H_
#define _RNNHOLDER_H_

#include "rnn.h"
#include "MultiImageRead.h"
class RnnHolder
{
public:
	RnnHolder();
	RnnHolder(string rnnParentPath);
	~RnnHolder();
	//输入为model2的十张图像的输出
	float runRnn(tensorflow::Tensor& tensor);
private:
	float runRnnThread2(int i, Tensor& inputTensor);
private:
	vector<rnn*> rnnHandle;

};

#endif