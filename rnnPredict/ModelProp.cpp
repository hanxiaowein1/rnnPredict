#include "ModelProp.h"

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

extern std::vector<std::string> split(std::string& s, char delimiter);

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

ModelProp::~ModelProp() {
	stopped.store(true);
	cv_task.notify_all();
	for (std::thread& thread : pool) {
		if (thread.joinable())
			thread.join();
	}
	//然后再把processTfModel1停掉
	tensor_queue_cv.notify_all();
}

void ModelProp::resizeImages(std::vector<cv::Mat>& imgs, int height, int width)
{
	if (imgs.size() == 0)
		return;
	if (imgs[0].rows != height)
	{
		for (auto& iter : imgs)
		{
			cv::resize(iter, iter, cv::Size(height, width));
		}
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
	resizeImages(imgs, inputProp.height, inputProp.width);
	clearResult();
	std::function<void(std::vector<cv::Mat>&)> mat2tensor_fun = std::bind(&ModelProp::convertMat2NeededDataInBatch, this, std::placeholders::_1);
	auto task = std::make_shared<std::packaged_task<void()>>
		(std::bind(&ModelProp::process2, this, std::ref(imgs), mat2tensor_fun));
	std::unique_lock<std::mutex> myGuard(m_lock);
	tasks.emplace(
		[task]() {
			(*task)();
		}
	);
	myGuard.unlock();
	cv_task.notify_one();
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
				if (!checkQueueEmpty() > 0 || stopped.load())
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

void ModelProp::createThreadPool()
{
	int num = idlThrNum;
	for (int size = 0; size < num; ++size)
	{   //初始化线程数量
		pool.emplace_back(
			[this]
			{ // 工作线程函数
				while (!this->stopped.load())
				{
					std::function<void()> task;
					{   // 获取一个待执行的 task
						std::unique_lock<std::mutex> lock{ this->m_lock };// unique_lock 相比 lock_guard 的好处是：可以随时 unlock() 和 lock()
						this->cv_task.wait(lock,
							[this] {
								return this->stopped.load() || !this->tasks.empty();
							}
						); // wait 直到有 task
						if (this->stopped.load() && this->tasks.empty())
							return;
						task = std::move(this->tasks.front()); // 取一个 task
						this->tasks.pop();
					}
					idlThrNum--;
					task();
					idlThrNum++;
				}
			}
			);
	}
}
