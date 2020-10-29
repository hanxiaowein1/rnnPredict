#include "TfModel2.h"

TfModel2::TfModel2(std::string iniPath, std::string group):TfBase(iniPath, group)
{
	inputProp.initByiniFile(iniPath, "Model2");
}

TfModel2::TfModel2(std::string group) : TfBase(group)
{
	inputProp.initByIniConfig("Model2");
}

void TfModel2::processInBatch(std::vector<cv::Mat> &imgs)
{
	vector<tensorflow::Tensor> tempTensors;
	output(imgs, tempTensors);
	vector<model2Result> tempResults = resultOutput(tempTensors);
	m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
}

vector<model2Result> TfModel2::resultOutput(vector<tensorflow::Tensor>& tensors)
{
	vector<model2Result> tempResults;
	vector<float> scores = resultOutput(tensors[0]);
	auto tensorValue = tensors[1].tensor<float, 2>();
	int tempSize = scores.size();
	const float* buffer_start = tensors[1].flat<float>().data();
	const float* buffer_end = tensors[1].flat<float>().data() + tempSize * 2048;
	for (int i = 0; i < scores.size(); i++)
	{
		model2Result result;
		result.score = scores[i];
		result.tensor.insert(result.tensor.end(), buffer_start + i * 2048, buffer_start + (i + 1) * 2048);
		tempResults.emplace_back(result);
	}
	return tempResults;
}

vector<float> TfModel2::resultOutput(tensorflow::Tensor& tensor)
{
	vector<float> scores;
	if (tensor.dims() != 2)
	{
		cout << "model2 output size should be two...\n";
		return scores;
	}
	auto scoreTensor = tensor.tensor<float, 2>();
	for (int i = 0; i < tensor.dim_size(0); i++)
	{
		float score = (scoreTensor(i, 0));
		scores.emplace_back(score);
	}
	return scores;
}

int TfModel2::processFirstDataInQueue()
{
	tensorflow::Tensor tensorInput = std::move(tensorQueue.front().second);
	int ret_size = tensorQueue.front().first;
	tensorQueue.pop();
	vector<tensorflow::Tensor> outputTensors;
	output(tensorInput, outputTensors);
	vector<model2Result> tempResults = resultOutput(outputTensors);
	m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
	return ret_size;
}

