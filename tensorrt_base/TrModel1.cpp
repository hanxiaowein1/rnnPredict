#include "TrModel1.h"
#include "model1.h"
#include "IniConfig.h"
#include <filesystem>

TrModel1::TrModel1(std::string iniPath, std::string group):TrBase(iniPath, group)
{
	inputProp.initByiniFile(iniPath, "Model1");
	unsigned long long memory = getMemory(iniPath, group);
	TrBase::build(memory, inputProp.batchsize);
}

TrModel1::TrModel1(std::string group) :TrBase(group)
{
	inputProp.initByIniConfig("Model1");
	unsigned long long memory = getMemory(group);
	initializeEnginePath();
	TrBase::build(memory, inputProp.batchsize);
}

void TrModel1::constructNetwork()
{
	mParser->registerInput(fileProp.inputName.c_str(), 
		nvinfer1::Dims3(inputProp.channel, inputProp.height, inputProp.width),
		nvuffparser::UffInputOrder::kNCHW);
	for (int i = 0; i < fileProp.outputNames.size() - 1; i++)
	{
		mParser->registerOutput(fileProp.outputNames[i].c_str());
	}
	//for (auto& elem : fileProp.outputNames)
	//{
	//	mParser->registerOutput(elem.c_str());
	//}
	//���������㣬һ��shuffle layer��һ��softmax layer
	mParser->parse(fileProp.filepath.c_str(), *mNetwork);
	ITensor* outputTensor = mNetwork->getOutput(1);
	auto shuffle_layer = mNetwork->addShuffle(*outputTensor);
	Permutation permutation;
	for (int i = 0; i < Dims::MAX_DIMS; i++)
	{
		permutation.order[i] = 0;
	}
	permutation.order[0] = 2;
	permutation.order[1] = 0;
	permutation.order[2] = 1;
	shuffle_layer->setFirstTranspose(permutation);
	auto softmax_layer = mNetwork->addSoftMax(*shuffle_layer->getOutput(0));
	softmax_layer->getOutput(0)->setName(fileProp.outputNames[2].c_str());
	//fileProp.outputNames.emplace_back("softmax/output");
	mNetwork->markOutput(*softmax_layer->getOutput(0));
}

bool TrModel1::processOutput(int size, vector<std::pair<float, cv::Mat>>& pairElems)
{
	float* output1 = static_cast<float*>(mBuffer->getHostBuffer(fileProp.outputNames[0]));
	float* output2 = static_cast<float*>(mBuffer->getHostBuffer(fileProp.outputNames[2]));
	if (size > inputProp.batchsize)
		return false;
	for (int i = 0; i < size; i++)
	{
		std::pair<float, cv::Mat> pairElem;
		pairElem.first = output1[i];
		cv::Mat temp(16, 16, CV_32FC1, output2 + 512 * i + 256);
		pairElem.second = temp.clone();
		pairElems.emplace_back(std::move(pairElem));
	}
	return true;
}

vector<model1Result> TrModel1::resultOutput(int size)
{
	vector<std::pair<float, cv::Mat>> pairElems;
	vector<model1Result> retResults;
	if (!processOutput(size, pairElems))
		return retResults;
	for (int i = 0; i < size; i++)
	{
		model1Result result;
		result.points = getRegionPoints2(pairElems[i].second, 0.7f);
		result.score = pairElems[i].first;
		retResults.emplace_back(result);
	}
	return retResults;
}

void TrModel1::processInBatch(std::vector<cv::Mat>& imgs)
{
	infer(imgs);
	vector<model1Result> tempResults = resultOutput(imgs.size());
	m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
}

int TrModel1::processFirstDataInQueue()
{
	vector<float> neededData = std::move(tensorQueue.front().second);
	int ret_size = tensorQueue.front().first;
	tensorQueue.pop();
	float* hostInputBuffer = static_cast<float*>((*mBuffer).getHostBuffer(fileProp.inputName));
	std::memcpy(hostInputBuffer, neededData.data(), neededData.size() * sizeof(float));
	mBuffer->copyInputToDevice();
	int tensorBatch = neededData.size() /
		(inputProp.height * inputProp.width * inputProp.channel);
	if (!mContext->execute(tensorBatch, mBuffer->getDeviceBindings().data()))
	{
		return ret_size;
	}
	mBuffer->copyOutputToHost();
	vector<model1Result> tempResults = resultOutput(tensorBatch);
	m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
	return ret_size;
}

void TrModel1::initializeEnginePath()
{
	if (IniConfig::instance().getIniString("TensorRT", "quantize") == "ON") {
		m_engine_path = "./engine/model1_fp16.engine";
	}
	else {
		m_engine_path = "./engine/model1.engine";
	}
}

bool TrModel1::checkModelChange()
{
	if (IniConfig::instance().getIniString("TrModel1", "engine_change_flag") == "False")
		return false;
	return true;
}