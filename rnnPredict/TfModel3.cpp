#include "TfModel3.h"

TfModel3::TfModel3(std::string iniPath, std::string group) :TfBase(iniPath, group)
{
	inputProp.initByiniFile(iniPath, "Model3");
}

void TfModel3::processInBatch(std::vector<cv::Mat>& imgs)
{
	vector<tensorflow::Tensor> tempTensors;
	output(imgs, tempTensors);
	vector<model3Result> tempResults = resultOutput(tempTensors[0]);
	m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
}

vector<model3Result> TfModel3::resultOutput(const tensorflow::Tensor& tensor)
{
	vector<model3Result> results;
	if (tensor.dims() != 2)
	{
		cout << "mdoel3 output size should be two\n";
		return results;
	}
	auto tensorValue = tensor.tensor<float, 2>();
	for (int i = 0; i < tensor.dim_size(0); i++)
	{
		model3Result result;
		result.scores[0] = tensorValue(i, 0);
		result.scores[1] = tensorValue(i, 1);
		result.scores[2] = tensorValue(i, 2);
		results.emplace_back(result);
	}
	return results;
}

void TfModel3::processFirstDataInQueue()
{
	tensorflow::Tensor tensorInput = std::move(tensorQueue.front());
	tensorQueue.pop();
	vector<tensorflow::Tensor> outputTensors;
	output(tensorInput, outputTensors);
	vector<model3Result> tempResults = resultOutput(outputTensors[0]);
	m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
}
