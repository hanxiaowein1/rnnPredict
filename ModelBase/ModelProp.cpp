#include "ModelProp.h"
#include "IniConfig.h"
#include "CommonFunction.h"

void ModelInputProp::initByiniFile(std::string iniPath, std::string group)
{
	char mpp_v[MAX_PATH];

	height = GetPrivateProfileInt(group.c_str(), "height", -1, iniPath.c_str());
	width = GetPrivateProfileInt(group.c_str(), "width", -1, iniPath.c_str());
	channel = GetPrivateProfileInt(group.c_str(), "channel", -1, iniPath.c_str());
	batchsize = GetPrivateProfileInt(group.c_str(), "batchsize", -1, iniPath.c_str());
	GetPrivateProfileString(group.c_str(), "mpp", "default", mpp_v, MAX_PATH, iniPath.c_str());

	mpp = std::stod(std::string(mpp_v));
}

void ModelInputProp::initByIniConfig(std::string group)
{
	height = IniConfig::instance().getIniInt(group, "height");
	width = IniConfig::instance().getIniInt(group, "width");
	channel = IniConfig::instance().getIniInt(group, "channel");
	batchsize = IniConfig::instance().getIniInt(group, "batchsize");
	mpp = IniConfig::instance().getIniDouble(group, "mpp");
}

void ModelFileProp::initByiniFile(std::string iniPath, std::string group)
{
	char inputName_v[MAX_PATH];
	char outputNames_v[MAX_PATH];
	char path_v[MAX_PATH];

	GetPrivateProfileString(group.c_str(), "input", "default", inputName_v, MAX_PATH, iniPath.c_str());
	GetPrivateProfileString(group.c_str(), "output", "default", outputNames_v, MAX_PATH, iniPath.c_str());
	GetPrivateProfileString(group.c_str(), "path", "default", path_v, MAX_PATH, iniPath.c_str());

	inputName = std::string(inputName_v);
	filepath = std::string(path_v);
	std::string compositeOutName = std::string(outputNames_v);

	outputNames = split(compositeOutName, ',');
}

void ModelFileProp::initByIniConfig(std::string group)
{
	inputName = IniConfig::instance().getIniString(group, "input");
	filepath = IniConfig::instance().getIniString(group, "path");
	std::string compositeOutName = IniConfig::instance().getIniString(group, "output");
	outputNames = split(compositeOutName, ',');
}

ModelProp::~ModelProp() {

}

void ModelProp::resizeImages(std::vector<cv::Mat>& imgs, int height, int width)
{
	if (imgs.size() == 0)
		return;
	if (imgs[0].rows != height)
	{
		for (int i = 0; i < imgs.size(); i++)
		{
			cv::resize(imgs[i], imgs[i], cv::Size(height, width));
		}
		//for (auto& iter : imgs)
		//{
		//	cv::resize(iter, iter, cv::Size(height, width));
		//}
	}
}

void ModelProp::process(std::vector<cv::Mat>& imgs)
{
	clearResult();
	resizeImages(imgs, inputProp.height, inputProp.width);
	int start = 0;
	for (int i = 0; i < imgs.size(); i = i + inputProp.batchsize)
	{
		auto iterBegin = imgs.begin() + start;
		std::vector<cv::Mat>::iterator iterEnd = imgs.end();
		if (iterBegin + inputProp.batchsize < iterEnd)
		{
			iterEnd = iterBegin + inputProp.batchsize;
			start = i + inputProp.batchsize;
		}
		std::vector<cv::Mat> tempImgs(iterBegin, iterEnd);
		processInBatch(tempImgs);
	}
}

void ModelProp::process2(std::vector<cv::Mat>& imgs, std::function<void(std::vector<cv::Mat>&)> inFunc)
{
	int start = 0;
	for (int i = 0; i < imgs.size(); i = i + inputProp.batchsize)
	{
		auto iterBegin = imgs.begin() + start;
		std::vector<cv::Mat>::iterator iterEnd = imgs.end();
		if (iterBegin + inputProp.batchsize < iterEnd)
		{
			iterEnd = iterBegin + inputProp.batchsize;
			start = i + inputProp.batchsize;
		}
		std::vector<cv::Mat> tempImgs(iterBegin, iterEnd);
		inFunc(tempImgs);
	}
}

void ModelProp::processDataConcurrency(std::vector<cv::Mat>& imgs)
{
	if (imgs.size() == 0)
		return;
	clearResult();
	std::function<void(std::vector<cv::Mat>&)> mat2tensor_fun = std::bind(&ModelProp::convertMat2NeededDataInBatch, this, std::placeholders::_1);
	auto task = std::make_shared<std::packaged_task<void()>>
		(std::bind(&ModelProp::process2, this, std::ref(imgs), mat2tensor_fun));
	std::unique_lock<std::mutex> task_lock(task_mutex);
	tasks.emplace(
		[task]() {
			(*task)();
		}
	);
	task_lock.unlock();
	task_cv.notify_one();
	//开始处理
	int loopTime = std::ceil(float(imgs.size()) / float(inputProp.batchsize));
	//判断队列是否为空
	for (int i = 0; i < loopTime; i++)
	{
		std::unique_lock<std::mutex> myGuard(queue_lock);
		if (!checkQueueEmpty())
		{
			//在这个里面处理队列中的第一个数据
			processFirstDataInQueue();
		}
		else
		{
			//等待
			tensor_queue_cv.wait(myGuard, [this] {
				if (!checkQueueEmpty() || stopped.load())
					return true;
				else
					return false;
				});
			if (stopped.load())
				return;
			processFirstDataInQueue();
		}
	}
}
