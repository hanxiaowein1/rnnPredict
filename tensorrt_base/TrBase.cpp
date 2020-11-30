#include "TrBase.h"
#include "IniConfig.h"
#include <fstream>
#include <memory>
#include <filesystem>
TrBase::TrBase(std::string iniPath, std::string group)
{
	fileProp.initByiniFile(iniPath, group);
}

TrBase::TrBase(std::string group)
{
	fileProp.initByIniConfig(group);
}

bool TrBase::transformInMemory(vector<cv::Mat>& imgs, float* dstPtr)
{
	if (imgs.size() == 0)
		return false;
	int width = imgs[0].cols;
	int height = imgs[0].rows;
	int channel = imgs[0].channels();
	for (int i = 0; i < imgs.size(); i++)
	{
		imgs[i].convertTo(imgs[i], CV_32F);
		imgs[i] = (imgs[i] / 255 - 0.5) * 2;
	}

	//注意顺序，是CHW，不是HWC
	for (int i = 0; i < imgs.size(); i++) {
		for (int c = 0; c < channel; c++) {
			for (int h = 0; h < height; h++) {
				float* linePtr = (float*)imgs[i].ptr(h);
				for (int w = 0; w < width; w++) {
					//换算地址
					dstPtr[i * height * width * channel + c * height * width + h * width + w] = *(linePtr + w * 3 + c);
				}
			}
		}
	}
	return true;
}

bool TrBase::processInput(vector<cv::Mat>& imgs)
{
	float* hostInputBuffer = static_cast<float*>((*mBuffer).getHostBuffer(fileProp.inputName));
	return transformInMemory(imgs, hostInputBuffer);
}

bool TrBase::infer(vector<cv::Mat>& imgs)
{
	processInput(imgs);
	mBuffer->copyInputToDevice();
	if (!mContext->execute(imgs.size(), mBuffer->getDeviceBindings().data()))
	{
		return false;
	}
	mBuffer->copyOutputToHost();
	return true;
}

bool TrBase::build(unsigned long long memory, int batchsize, bool set_fp16_flag)
{
	mBuilder.reset(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));//这里要考虑同时用一个全局变量初始化多个模型会不会出错
	if (!mBuilder)
	{
		return false;
	}
	mNetwork.reset(mBuilder->createNetwork());
	if (!mNetwork)
	{
		return false;
	}
	mConfig.reset(mBuilder->createBuilderConfig());
	if (!mConfig)
	{
		return false;
	}
	mParser.reset(nvuffparser::createUffParser());
	if (!mParser)
	{
		return false;
	}
	memory = memory * (1 << 30);
	mBuilder->setMaxWorkspaceSize(memory);
	constructNetwork();
	mBuilder->setMaxBatchSize(batchsize);
	if (set_fp16_flag)
	{
		mBuilder->setFp16Mode(true);
	}
	//mBuilder->setHalf2Mode(true);


	//mConfig->setMaxWorkspaceSize(memory);
	//mConfig->setFlag(BuilderFlag::kGPU_FALLBACK);
	//mConfig->setFlag(BuilderFlag::kFP16);

	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mBuilder->buildCudaEngine(*mNetwork), samplesCommon::InferDeleter());

	//samplesCommon::enableDLA(mBuilder.get(), mConfig.get(), -1);
	//mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mBuilder->buildEngineWithConfig(*mNetwork, *mConfig), samplesCommon::InferDeleter());
	//保存一下engine文件

	if (!mEngine)
	{
		return false;
	}
	return buildContextAndBuffer();
}

bool TrBase::buildContextAndBuffer()
{
	mContext.reset(mEngine->createExecutionContext());
	if (!mContext)
	{
		return false;
	}
	mBuffer = new samplesCommon::BufferManager(mEngine, inputProp.batchsize);
	if (!mBuffer)
	{
		return false;
	}
	return true;
}

bool TrBase::build(unsigned long long memory, int batchsize)
{
	bool ret = false;
	//std::cout << IniConfig::instance().getIniString("TensorRT", "quantize") << std::endl;
	if (IniConfig::instance().getIniString("TensorRT", "quantize") == "ON") {
		//判断是否更改了模型配置，或者如果没有该文件，则build，并保存engine文件
		if (checkModelChange() || !checkEngineExists()) {
			ret =  build(memory, batchsize, true);
		}
	}
	else {
		//判断是否更改了模型配置，或者如果没有该文件，则build，并保存engine文件
		if (checkModelChange() || !checkEngineExists()) {
			ret =  build(memory, batchsize, false);
		}
	}
	if (ret) {
		saveEngine(m_engine_path);
		return ret;
	}
	buildEngine();
	return true;
}

unsigned long long TrBase::getMemory(std::string iniPath, std::string group)
{
	return GetPrivateProfileInt(group.c_str(), "memory", 3, iniPath.c_str());
}

unsigned long long TrBase::getMemory(std::string group)
{
	//return GetPrivateProfileInt(group.c_str(), "memory", 3, iniPath.c_str());
	return IniConfig::instance().getIniInt(group, "memory");
}

bool TrBase::checkQueueEmpty()
{
	if (tensorQueue.empty())
		return true;
	else
		return false;
}

void TrBase::convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs)
{
	int size = imgs.size();
	if (size == 0)
		return;
	resizeImages(imgs, inputProp.height, inputProp.width);
	int height = imgs[0].rows;
	int width = imgs[0].cols;
	int channel = imgs[0].channels();
	vector<float> neededData(height * width * channel * size);
	transformInMemory(imgs, neededData.data());
	//将其塞到队列里
	std::unique_lock<std::mutex> myGuard(queue_lock);
	std::pair<int, std::vector<float>> temp_elem;
	temp_elem.first = imgs.size();
	temp_elem.second = std::move(neededData);
	tensorQueue.emplace(std::move(temp_elem));
	myGuard.unlock();
	//通过条件变量通知另一个等待线程：队列里有数据了！
	tensor_queue_cv.notify_one();
}

void TrBase::saveEngine(std::string engine_path)
{
	if (mEngine)
	{
		std::filesystem::path sf_engine_path(engine_path);
		std::string engine_parent_path = sf_engine_path.parent_path().string();
		if (!std::filesystem::exists(engine_parent_path))
			std::filesystem::create_directories(engine_parent_path);
		IHostMemory* serialized_model = mEngine->serialize();		
		std::ofstream serialize_output_stream(engine_path.c_str(), std::fstream::out | std::fstream::binary);
		if (serialize_output_stream) {
			serialize_output_stream.write((char*)serialized_model->data(), serialized_model->size());
		}
		serialize_output_stream.close();
		serialized_model->destroy();
	}
}

void TrBase::buildEngine()
{
	buildEngine(m_engine_path);
}

void TrBase::buildEngine(std::string engine_path)
{
	std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
	std::streamsize size = file.tellg();
	//std::unique_ptr<char[]> buffer(new char[size]);
	char* buffer = new char[size];
	file.seekg(0, std::ios::beg);
	if (!file.read(buffer, size)) {
		std::cout << "read file to buffer failed" << endl;
	}
	IRuntime* runtime = createInferRuntime(gLogger);
	//mEngine.reset(runtime->deserializeCudaEngine(buffer.get(), size, nullptr));
	//mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mBuilder->buildCudaEngine(*mNetwork), samplesCommon::InferDeleter());
	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer, size, nullptr), samplesCommon::InferDeleter());
	buildContextAndBuffer();
	delete[]buffer;
}

bool TrBase::checkEngineExists()
{
	if (std::filesystem::exists(m_engine_path))
		return true;
	return false;
}