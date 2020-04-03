#pragma once
#ifndef _RNN_H_
#define _RNN_H_


#include "model.h"
class rnn : public model
{
public:
	rnn();
	rnn(modelConfig config, char* buffer, int size);
	vector<float> rnnProcess(Tensor& tensor);
};

#endif // !_RNN_H_