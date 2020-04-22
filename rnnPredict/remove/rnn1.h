#pragma once
#ifndef _RNN_1_
#define _RNN_1_

#include "model.h"
class rnn1 : public model
{
	rnn1();
	rnn1(modelConfig config, char* buffer, int size);
	vector<float> process(Tensor& tensor);
};

#endif