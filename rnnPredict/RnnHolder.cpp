#include "RnnHolder.h"
#include <algorithm>
#include <numeric>
RnnHolder::RnnHolder()
{
}

RnnHolder::RnnHolder(string iniPath)
{
	rnnConfig(iniPath);
}

RnnHolder::~RnnHolder()
{
	for (int i = 0; i < rnnHandle.size(); i++) {
		delete rnnHandle[i];
	}
}

void RnnHolder::rnnConfig(string iniPath)
{
	vector<string> groups{"TfRnn1","TfRnn2", "TfRnn3", "TfRnn4", "TfRnn5", "TfRnn6"};
	for (auto iter : groups)
	{
		TfRnn* rnn_obj = new TfRnn(iniPath, iter);
		rnnHandle.emplace_back(rnn_obj);
	}

}

float RnnHolder::runRnn(tensorflow::Tensor& tensor)
{
	vector<tensorflow::Tensor> rnnInputTensor;
	tensorflow::Tensor tensor10 = tensor.Slice(0, 10);
	tensorflow::Tensor tensor20 = tensor.Slice(0, 20);
	rnnInputTensor.emplace_back(tensor10);
	rnnInputTensor.emplace_back(tensor20);
	rnnInputTensor.emplace_back(tensor);

	vector<std::future<float>> rnnResults(rnnHandle.size());
	for (int i = 0; i < rnnHandle.size(); i++)
	{
		rnnResults[i] = std::async(&RnnHolder::runRnnThread2, this, i, std::ref(rnnInputTensor[i / 2]));
	}
	vector<float> rnnResults_f;
	for (int i = 0; i < rnnResults.size(); i++)
	{
		rnnResults_f.emplace_back(rnnResults[i].get());
	}
	cout << "rnnResults_f: ";
	for (int i = 0; i < rnnResults_f.size(); i++)
	{
		cout << rnnResults_f[i] << " ";
	}
	cout << endl;
	if (rnnResults_f.size() != 6)
		return -1;
	//先求6个数的平均值
	float sum_6 = std::accumulate(rnnResults_f.begin(), rnnResults_f.end(), 0.0f);
	float avg_6 = sum_6 / rnnResults_f.size();
	float avg2_1 = std::accumulate(rnnResults_f.begin(), rnnResults_f.begin() + 2, 0.0f) / 2;//前两个平均值
	float avg2_2 = std::accumulate(rnnResults_f.begin() + 2, rnnResults_f.begin() + 4, 0.0f) / 2;//...
	float avg2_3 = std::accumulate(rnnResults_f.begin() + 4, rnnResults_f.end(), 0.0f) / 2;//...
	vector<float> avg3_total{ avg2_1, avg2_2, avg2_3 };
	float max_3 = *std::max_element(avg3_total.begin(), avg3_total.end());
	float min_3 = *std::min_element(avg3_total.begin(), avg3_total.end());

	float sum_3 = std::accumulate(avg3_total.begin(), avg3_total.end(), 0.0f);
	float avg_3 = sum_3 / avg3_total.size();
	float accum = 0.0;
	std::for_each(avg3_total.begin(), avg3_total.end(), [&](const float d) {
		accum += (d - avg_3) * (d - avg_3);
		});
	accum = accum / avg3_total.size();//方差
	float std_3 = std::pow(accum, 0.5f);

	float retScore = 0.0f;
	if (std_3 < 0.15f)
	{
		if (avg_6 < 0.15f)
			retScore = min_3;
		else
			retScore = max_3;
	}
	else
		retScore = avg_6;
	return retScore;
}

float RnnHolder::runRnnThread2(int i, tensorflow::Tensor& inputTensor)
{
	//对inputTensor要转为1*?*?
	if (inputTensor.dims() != 2)
	{
		cout << "runRnnThread2: inputTensor dims should be 2\n";
		return -1;
	}
	tensorflow::Tensor tem_tensor_res(
		tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({ 1, inputTensor.dim_size(0), inputTensor.dim_size(1) }));
	std::memcpy(tem_tensor_res.flat<float>().data(), inputTensor.flat<float>().data(),
		inputTensor.dim_size(0) * inputTensor.dim_size(1) * sizeof(float));
	vector<float> score = rnnHandle[i]->rnnProcess(tem_tensor_res);
	return score[0];
}