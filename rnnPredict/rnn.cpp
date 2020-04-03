#include "rnn.h"


rnn::rnn()
{
}

rnn::rnn(modelConfig config, char* buffer, int size) :model(config, buffer, size)
{

}


vector<float> rnn::rnnProcess(Tensor& tensor)
{
	vector<float> score;
	if (tensor.dims() != 2)
	{
		cout << "rnn model output tensor dims should be 2...\n";
		return score;
	}
	auto scoreTensor = tensor.tensor<float, 2>();
	for (int i = 0; i < tensor.dim_size(1); i++)
	{
		score.emplace_back(scoreTensor(0, i));
	}
	return score;
	//return vector<float>();
}
