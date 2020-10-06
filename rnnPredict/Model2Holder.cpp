#include "Model2Holder.h"
#include "IniConfig.h"
Model2Holder::Model2Holder()
{
}

Model2Holder::Model2Holder(std::string iniPath)
{
	if (IniConfig::instance().getIniString("TensorRT", "USE_TR") == "ON")
		use_tr = true;
	else if (IniConfig::instance().getIniString("TensorRT", "USE_TR") == "OFF")
		use_tr = false;
	else
		use_tr = false;
	model2Config(iniPath);
}

Model2Holder::~Model2Holder()
{
	stopped.store(true);
	data_cv.notify_all();
	task_cv.notify_all();
	for (std::thread& thread : pool) {
		if (thread.joinable())
			thread.join();
	}
}

void Model2Holder::initPara(MultiImageRead& mImgRead)
{
	mImgRead.getSlideHeight(slideHeight);
	mImgRead.getSlideWidth(slideWidth);
	mImgRead.getSlideMpp(slideMpp);

	model1Height = IniConfig::instance().getIniInt("Model1", "height");
	model1Width = IniConfig::instance().getIniInt("Model1", "width");
	model1Mpp = IniConfig::instance().getIniDouble("Model1", "mpp");
}

void Model2Holder::model2Config(std::string iniPath)
{
	//model2Handle = std::make_unique<TrModel2>(iniPath, "TrModel2");
	//model2Handle = std::make_unique<TfModel2>(iniPath, "TfModel2");
	//model2Handle = std::make_unique<TfModel2>("TfModel2");
	//model2Handle->createThreadPool();
	//model2Mpp = model2Handle->inputProp.mpp;
	//model2Height = model2Handle->inputProp.height;
	//model2Width = model2Handle->inputProp.width;

	if (!use_tr)
	{
		model2Handle.first = std::make_unique<TfModel2>("TfModel2");
		model2Handle.first->createThreadPool();
		model2Mpp = model2Handle.first->inputProp.mpp;
		model2Height = model2Handle.first->inputProp.height;
		model2Width = model2Handle.first->inputProp.width;
	}
	else
	{
		model2Handle.second = std::make_unique<TrModel2>("TrModel2");
		model2Handle.second->createThreadPool();
		model2Mpp = model2Handle.second->inputProp.mpp;
		model2Height = model2Handle.second->inputProp.height;
		model2Width = model2Handle.second->inputProp.width;
	}
}

void Model2Holder::createThreadPool(int threadNum)
{
	idlThrNum = threadNum;
	totalThrNum = threadNum;
	for (int size = 0; size < totalThrNum; ++size)
	{   //初始化线程数量
		pool.emplace_back(
			[this]
			{ // 工作线程函数
				while (!this->stopped.load())
				{
					std::function<void()> task;
					{   // 获取一个待执行的 task
						std::unique_lock<std::mutex> lock{ this->task_mutex };// unique_lock 相比 lock_guard 的好处是：可以随时 unlock() 和 lock()
						this->task_cv.wait(lock,
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

void Model2Holder::pushData(MultiImageRead& mImgRead)
{
	std::vector<std::pair<cv::Rect, cv::Mat>> tempRectMats;
	while (mImgRead.popQueue(tempRectMats)) {
		std::unique_lock<std::mutex> data_lock(data_mutex);
		for (auto iter = tempRectMats.begin(); iter != tempRectMats.end(); iter++) {
			data_queue.emplace(std::move(*iter));
		}
		data_lock.unlock();
		data_cv.notify_one();
		tempRectMats.clear();
	}
}

void Model2Holder::popQueueWithoutLock(vector<std::pair<cv::Rect, cv::Mat>>& rectMats)
{
	int size = data_queue.size();
	for (int i = 0; i < size; i++)
	{
		rectMats.emplace_back(std::move(data_queue.front()));
		data_queue.pop();
	}
}

bool Model2Holder::popData(std::vector<std::pair<cv::Rect, cv::Mat>>& rectMats)
{
	std::unique_lock<std::mutex> data_lock(data_mutex);
	if (data_queue.size() > 0)
	{
		popQueueWithoutLock(rectMats);
		data_lock.unlock();
		return true;
	}
	else
	{
		data_lock.unlock();
		//取得tasks的锁，检查是否还有任务
		std::unique_lock<std::mutex> task_lock(task_mutex);
		if (tasks.size() > 0)
		{
			task_lock.unlock();
			//证明有task，那么data_mutex再次锁上，因为一定会有数据唤醒它
			data_lock.lock();
			data_cv.wait(data_lock, [this] {
				if (data_queue.size() > 0 || stopped.load()) {
					return true;
				}
				else {
					return false;
				}
				});
			if (stopped.load())
				return false;
			popQueueWithoutLock(rectMats);
			data_lock.unlock();
			return true;
		}
		else
		{
			task_lock.unlock();
			//如果没有task，那么要检查是否有线程在运行
			if (idlThrNum.load() == totalThrNum.load())
			{
				//如果没有线程在运行，那么在看看队列是否有元素（万一在判断的时候进入了队列了呢？）
				data_lock.lock();//其实不用加锁，因为没有线程在运行，肯定不会占用锁了
				popQueueWithoutLock(rectMats);
				data_lock.unlock();
				if (rectMats.size() == 0)
					return false;
				return true;
			}
			else
			{
				//如果有线程在运行，那么在锁住队列，等待人来唤醒
				data_lock.lock();
				data_cv.wait_for(data_lock, 1000ms, [this] {
					if (data_queue.size() > 0 || stopped.load()) {
						return true;
					}
					else {
						return false;
					}
					});
				if (stopped.load())
					return false;
				data_lock.unlock();
				return popData(rectMats);
			}
		}
	}
}

void Model2Holder::sortResultsByScore(std::vector<regionResult>& results)
{
	auto lambda = [](regionResult result1, regionResult result2)->bool {
		if (result1.result.score > result2.result.score)
			return true;
		return false;
	};
	std::sort(results.begin(), results.end(), lambda);
}

void Model2Holder::model2Process(std::vector<cv::Mat>& imgs, std::vector<model2Result>& results)
{
	//model2Handle->processDataConcurrency(imgs);
	//results = model2Handle->m_results;
	if (!use_tr)
	{
		model2Handle.first->processDataConcurrency(imgs);
		results = model2Handle.first->m_results;
	}
	else
	{
		model2Handle.second->processDataConcurrency(imgs);
		results = model2Handle.second->m_results;
	}
}

void Model2Holder::readImageInOrder(std::vector<cv::Rect> rects, MultiImageRead& mImgRead, std::vector<cv::Mat>& imgs)
{
	startRead(rects, mImgRead);
	vector<std::pair<cv::Rect, cv::Mat>> rectMats;
	vector<std::pair<cv::Rect, cv::Mat>> tmpRectMats;
	while (popData(tmpRectMats)) {
		for (auto iter = tmpRectMats.begin(); iter != tmpRectMats.end(); iter++) {
			rectMats.emplace_back(std::move(*iter));
		}
		tmpRectMats.clear();
	}
	//一个重点的错误！！rnn进入的tensor应该是从大到小的
	//无需flags，因为去重之后坐标不可能相同
	vector<bool> flags(rects.size(), false);
	vector<std::pair<cv::Rect, cv::Mat>> rectMats2(rectMats.size());

	for (int i = 0; i < rects.size(); i++)
	{
		for (int j = 0; j < rects.size(); j++)
		{
			if (rectMats[i].first.x == rects[j].x && rectMats[i].first.y == rects[j].y)
			{
				rectMats2[j] = std::move(rectMats[i]);
				break;
			}
		}
	}
	for (int i = 0; i < rectMats.size(); i++) {
		imgs.emplace_back(std::move(rectMats2[i].second));
	}
}

void Model2Holder::startRead(std::vector<cv::Rect> rects, MultiImageRead& mImgRead)
{
	mImgRead.setRects(rects);

	for (int i = 0; i < totalThrNum; i++)
	{
		auto task = std::make_shared<std::packaged_task<void()>>
			(std::bind(&Model2Holder::pushData, this, std::ref(mImgRead)));
		std::unique_lock<std::mutex> task_lock(task_mutex);
		tasks.emplace(
			[task]() {
				(*task)();
			}
		);
		task_lock.unlock();
		task_cv.notify_one();
	}
}

void Model2Holder::runModel2(MultiImageRead& mImgRead, std::vector<regionResult>& rResults)
{
	initPara(mImgRead);
	//从rResults里面挑选进入model2的框
	//按照分数从大到小排序
	int m2MinNum = 600;
	int m2MaxNum = 1200;

	sortResultsByScore(rResults);
	std::vector<cv::Rect> rects;
	int placeStop = 0;

	//我可以一开始就将model1定位点越界的全部干掉啊...
	//先copy一个副本，清除rResults，然后再从副本里面挑合理的放到rResults里面
	std::vector<regionResult> rResultsCP = rResults;
	rResults.clear();
	for (auto iter = rResultsCP.begin(); iter != rResultsCP.end(); iter++)
	{
		regionResult result;
		result.point = iter->point;
		result.result.score = iter->result.score;
		result.result.points.emplace_back(iter->result.points[0]);
		for (int i = 1; i < iter->result.points.size(); i++) {
			//cv::Point point = iter->point;
			cv::Rect rect;
			rect.x = iter->point.x;
			rect.y = iter->point.y;
			rect.x = rect.x + iter->result.points[i].x * float(model1Mpp / slideMpp) - model2Width * float(model2Mpp / slideMpp) / 2;
			rect.y = rect.y + iter->result.points[i].y * float(model1Mpp / slideMpp) - model2Height * float(model2Mpp / slideMpp) / 2;
			if (rect.x < 0 || rect.y < 0)
				continue;
			rect.width = model2Width * float(model2Mpp / slideMpp);
			rect.height = model2Height * float(model2Mpp / slideMpp);
			if ((rect.x + rect.width) > slideWidth || (rect.y + rect.height) > slideHeight)
				continue;
			result.result.points.emplace_back(iter->result.points[i]);
		}
		rResults.emplace_back(result);
	}

	for (auto iter = rResults.begin(); iter != rResults.end(); iter++)
	{
		//先将大于0.5的全部送到里面
		if (iter->result.score > 0.5) {
			if (rects.size() > m2MaxNum)
				break;
			placeStop = iter - rResults.begin();
			int allocSize = 0;
			for (int i = 1; i < iter->result.points.size(); i++) {
				cv::Rect rect;
				rect.x = iter->point.x
					+ iter->result.points[i].x * float(model1Mpp / slideMpp)
					- model2Width * float(model2Mpp / slideMpp) / 2;
				rect.y = iter->point.y
					+ iter->result.points[i].y * float(model1Mpp / slideMpp)
					- model2Height * float(model2Mpp / slideMpp) / 2;
				rect.width = model2Width * float(model2Mpp / slideMpp);
				rect.height = model2Height * float(model2Mpp / slideMpp);
				rects.emplace_back(rect);
			}
		}
		else {
			if (rects.size() < m2MinNum) {
				placeStop = iter - rResults.begin();
				//在向里面送
				int allocSize = 0;
				for (int i = 1; i < iter->result.points.size(); i++) {
					cv::Rect rect;
					rect.x = iter->point.x
						+ iter->result.points[i].x * float(model1Mpp / slideMpp)
						- model2Width * float(model2Mpp / slideMpp) / 2;
					rect.y = iter->point.y
						+ iter->result.points[i].y * float(model1Mpp / slideMpp)
						- model2Height * float(model2Mpp / slideMpp) / 2;
					rect.width = model2Width * float(model2Mpp / slideMpp);
					rect.height = model2Height * float(model2Mpp / slideMpp);
					rects.emplace_back(rect);
				}
			}
			else {
				break;
			}
		}
	}
	//placeStop是以0为起点的，所以计数时，要将其加一
	placeStop++;
	mImgRead.setReadLevel(0);//永恒不变，model2一直都是从level0上进行读取
	mImgRead.setQueueMaxNum(rects.size());
	startRead(rects, mImgRead);

	std::vector<std::pair<cv::Rect, cv::Mat>> rectMats;
	std::vector<PointScore> model2PS;
	while (popData(rectMats))
	{
		std::vector<cv::Mat> imgs;
		std::vector<cv::Rect> tmpRects;
		for (auto iter = rectMats.begin(); iter != rectMats.end(); iter++) {
			imgs.emplace_back(std::move(iter->second));//用move语义将其放到新的里面
			tmpRects.emplace_back(std::move(iter->first));
		}

		//model2Handle->processDataConcurrency(imgs);
		//std::vector<model2Result> tempResults = model2Handle->m_results;
		std::vector<model2Result> tempResults;
		if (!use_tr)
		{
			model2Handle.first->processDataConcurrency(imgs);
			tempResults = model2Handle.first->m_results;
		}
		else
		{
			model2Handle.second->processDataConcurrency(imgs);
			tempResults = model2Handle.second->m_results;
		}
		for (int i = 0; i < tmpRects.size(); i++) {
			PointScore ps;
			cv::Point point;
			point.x = tmpRects[i].x;
			point.y = tmpRects[i].y;
			ps.point = point;
			ps.score = tempResults[i].score;
			model2PS.emplace_back(ps);
		}
		rectMats.clear();
	}
	std::vector<std::vector<bool>> inFlag(placeStop);
	for (int i = 0; i < placeStop; i++) {
		int allocSize = rResults[i].result.points.size() - 1;
		if (allocSize > 0) {
			inFlag[i].resize(allocSize, false);
			rResults[i].score2.resize(allocSize);
		}
	}
	//考虑将model2PS放到rResults中
	for (auto iter = model2PS.begin(); iter != model2PS.end(); iter++) {
		cv::Point childPoint = iter->point;//这个是model2的左上角点
		for (int i = 0; i < placeStop; i++) {
			for (int j = 1; j < rResults[i].result.points.size(); j++) {
				cv::Point fatherPoint = rResults[i].result.points[j];
				fatherPoint.x = rResults[i].point.x + fatherPoint.x * float(model1Mpp / slideMpp) - model2Height * float(model2Mpp / slideMpp) / 2;//这个就是model2的全局定位点
				fatherPoint.y = rResults[i].point.y + fatherPoint.y * float(model1Mpp / slideMpp) - model2Width * float(model2Mpp / slideMpp) / 2;
				if (fatherPoint == childPoint) {
					//找到两个点是相同的且score2标志位不为true
					if (!inFlag[i][j - 1]) {
						//将model2PS的分数放入其中
						rResults[i].score2[j - 1] = iter->score;
					}
				}
			}
		}
	}
}