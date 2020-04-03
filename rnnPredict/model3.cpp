#include "model3.h"

model3::model3()
{
}

model3::model3(modelConfig config, char* buffer, int size) :model(config, buffer, size)
{

}

vector<model3Result> model3::resultOutput(const Tensor& tensor)
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

vector<model3Result> model3::model3Process(vector<cv::Mat>& imgs)
{
	vector<model3Result> results;
	if (imgs.size() == 0)
	{
		return results;
	}
	int start = 0;
	for (int i = 0; i < imgs.size(); i = i + batchsize)
	{
		auto iterBegin = imgs.begin() + start;
		auto iterEnd = imgs.end();
		if (iterBegin + batchsize > iterEnd)
		{
			vector<cv::Mat> tempImgs(iterBegin, iterEnd);
			vector<tensorflow::Tensor> tempTensors;
			output(tempImgs, tempTensors);
			vector<model3Result> tempResults = resultOutput(tempTensors[0]);
			results.insert(results.end(), tempResults.begin(), tempResults.end());
		}
		else
		{
			iterEnd = iterBegin + batchsize;
			vector<cv::Mat> tempImgs(iterBegin, iterEnd);
			vector<tensorflow::Tensor> tempTensors;
			output(tempImgs, tempTensors);
			vector<model3Result> tempResults = resultOutput(tempTensors[0]);
			results.insert(results.end(), tempResults.begin(), tempResults.end());
			start = i + batchsize;
		}
	}
	return results;
}