#include "TrBase.h"
#include "IniConfig.h"
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

	//ע��˳����CHW������HWC
	for (int i = 0; i < imgs.size(); i++) {
		for (int c = 0; c < channel; c++) {
			for (int h = 0; h < height; h++) {
				float* linePtr = (float*)imgs[i].ptr(h);
				for (int w = 0; w < width; w++) {
					//�����ַ
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

bool TrBase::build(unsigned long long memory, int batchsize)
{
	mBuilder.reset(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));//����Ҫ����ͬʱ��һ��ȫ�ֱ�����ʼ�����ģ�ͻ᲻�����
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
	constructNetwork();
	mBuilder->setMaxBatchSize(batchsize);
	memory = memory * (1 << 30);
	mConfig->setMaxWorkspaceSize(memory);
	mConfig->setFlag(BuilderFlag::kGPU_FALLBACK);

	samplesCommon::enableDLA(mBuilder.get(), mConfig.get(), -1);
	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mBuilder->buildEngineWithConfig(*mNetwork, *mConfig), samplesCommon::InferDeleter());
	if (!mEngine)
	{
		return false;
	}
	mContext.reset(mEngine->createExecutionContext());
	if (!mContext)
	{
		return false;
	}
	mBuffer = new samplesCommon::BufferManager(mEngine, batchsize);
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
	//��������������
	std::unique_lock<std::mutex> myGuard(queue_lock);
	tensorQueue.emplace(std::move(neededData));
	myGuard.unlock();
	//ͨ����������֪ͨ��һ���ȴ��̣߳��������������ˣ�
	tensor_queue_cv.notify_one();
}