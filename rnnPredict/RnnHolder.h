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
	//����Ϊmodel2��ʮ��ͼ������
	float runRnn(tensorflow::Tensor& tensor);
private:
	float runRnnThread2(int i, Tensor& inputTensor);
private:
	vector<rnn*> rnnHandle;

};

#endif