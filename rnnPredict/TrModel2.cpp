#include "TrModel2.h"

TrModel2::TrModel2(std::string iniPath, std::string group) : TrBase(iniPath, group)
{
	inputProp.initByiniFile(iniPath, "Model2");
	//这个时候Model1已经构造完成，开始配置mParam
	//paramConfig({ inputName }, outputNames, { channel, height, width }, batchsize);
	unsigned long long memory = getMemory(iniPath, group);
	TrBase::build(memory, inputProp.batchsize);
}

void TrModel2::constructNetwork()
{
	mParser->registerInput(fileProp.inputName.c_str(),
		nvinfer1::Dims3(inputProp.channel, inputProp.height, inputProp.width), 
		nvuffparser::UffInputOrder::kNCHW);
	for (auto& elem : fileProp.outputNames)
	{
		mParser->registerOutput(elem.c_str());
	}
	mParser->parse(fileProp.filepath.c_str(), *mNetwork, nvinfer1::DataType::kFLOAT);
}

vector<model2Result> TrModel2::resultOutput(int size)
{
	vector<model2Result> tempResults;
	vector<float> scores;
	if (!processOutput(size, scores))
		return tempResults;
	vector<vector<float>> tensors;
	if (!processOutput2(size, tensors))
		return tempResults;
	for (int i = 0; i < size; i++)
	{
		model2Result result;
		result.score = scores[i];
		result.tensor = tensors[i];
		tempResults.emplace_back(result);
	}
	return tempResults;
}

bool TrModel2::processOutput2(int size, vector<vector<float>>& tensors)
{
	float* output2 = static_cast<float*>(mBuffer->getHostBuffer(fileProp.outputNames[1]));
	for (int i = 0; i < size; i++)
	{
		vector<float> tmpTensor;
		tmpTensor.insert(tmpTensor.end(), output2 + i * 2048, output2 + (i + 1) * 2048);
		tensors.emplace_back(tmpTensor);
	}
	return true;
}

bool TrModel2::processOutput(int size, vector<float>& scores)
{
	float* output1 = static_cast<float*>(mBuffer->getHostBuffer(fileProp.outputNames[0]));
	if (size > inputProp.batchsize)
		return false;
	for (int i = 0; i < size; i++)
	{
		float score = output1[i];
		scores.emplace_back(score);
	}
	return true;
}

void TrModel2::processInBatch(std::vector<cv::Mat>& imgs)
{
	infer(imgs);
	vector<model2Result> tempResults = resultOutput(imgs.size());
	m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
}

void TrModel2::processFirstDataInQueue()
{
	vector<float> neededData = std::move(tensorQueue.front());
	tensorQueue.pop();
	float* hostInputBuffer = static_cast<float*>((*mBuffer).getHostBuffer(fileProp.inputName));
	std::memcpy(hostInputBuffer, neededData.data(), neededData.size() * sizeof(float));
	mBuffer->copyInputToDevice();
	int tensorBatch = neededData.size() /
		(inputProp.height * inputProp.width * inputProp.channel);
	if (!mContext->execute(tensorBatch, mBuffer->getDeviceBindings().data()))
	{
		return;
	}
	mBuffer->copyOutputToHost();
	vector<model2Result> tempResults = resultOutput(tensorBatch);
	m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
}