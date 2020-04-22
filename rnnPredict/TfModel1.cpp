#include "TfModel1.h"

extern std::vector<cv::Point> getRegionPoints2(cv::Mat& mask, float threshold);

TfModel1::TfModel1(string iniPath, string group):
	TfBase(iniPath, group)
{
	inputProp.initByiniFile(iniPath, "Model1");
}

TfModel1::~TfModel1()
{

}

vector<model1Result> TfModel1::resultOutput(vector<tensorflow::Tensor>& tensors)
{
	vector<model1Result> retResults;
	if (tensors.size() != 2)
	{
		cout << "model1Base::output: tensors size should be 2\n";
		return retResults;
	}
	auto scores = tensors[0].tensor<float, 2>();
	for (int i = 0; i < tensors[0].dim_size(0); i++)
	{
		model1Result result;
		cv::Mat dst2;
		TensorToMat(tensors[1].Slice(i, i + 1), &dst2);
		result.points = getRegionPoints2(dst2, 0.7);
		result.score = scores(i, 0);
		retResults.emplace_back(result);
	}
	return retResults;
}

void TfModel1::TensorToMat(tensorflow::Tensor mask, cv::Mat* dst)
{
	float* data = new float[(mask.dim_size(1)) * (mask.dim_size(2))];
	auto output_c = mask.tensor<float, 4>();
	//cout << "data 1 :" << endl;
	for (int j = 0; j < mask.dim_size(1); j++) {
		for (int k = 0; k < mask.dim_size(2); k++) {
			data[j * mask.dim_size(1) + k] = output_c(0, j, k, 1);
		}
	}
	cv::Mat myMat = cv::Mat(mask.dim_size(1), mask.dim_size(2), CV_32FC1, data);
	*dst = myMat.clone();
	delete[]data;
}

void TfModel1::processInBatch(std::vector<cv::Mat>& imgs)
{
	vector<tensorflow::Tensor> tempTensors;
	output(imgs, tempTensors);
	vector<model1Result> tempResults = resultOutput(tempTensors);
	m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
}

void TfModel1::processFirstDataInQueue()
{
	tensorflow::Tensor tensorInput = std::move(tensorQueue.front());
	tensorQueue.pop();
	vector<tensorflow::Tensor> outputTensors;
	output(tensorInput, outputTensors);
	vector<model1Result> tempResults = resultOutput(outputTensors);
	m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
}
