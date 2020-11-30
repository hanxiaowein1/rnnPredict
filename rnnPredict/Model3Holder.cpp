#include "Model3Holder.h"
#include "IniConfig.h"
Model3Holder::Model3Holder()
{}

Model3Holder::Model3Holder(std::string iniPath)
{
	model3Config(iniPath);
}

Model3Holder::~Model3Holder()
{
	stopped.store(true);
	data_cv.notify_all();
	task_cv.notify_all();
	for (std::thread& thread : pool) {
		if (thread.joinable())
			thread.join();
	}
}

void Model3Holder::initPara(MultiImageRead& mImgRead)
{
	mImgRead.getSlideHeight(slideHeight);
	mImgRead.getSlideWidth(slideWidth);
	mImgRead.getSlideMpp(slideMpp);
}

void Model3Holder::model3Config(std::string iniPath)
{
	//model3Handle = std::make_unique<TfModel3>(iniPath, "TfModel3");
	model3Handle = std::make_unique<TfModel3>("TfModel3");
	model3Handle->createThreadPool();
	model3Mpp = IniConfig::instance().getIniDouble("Model3", "mpp");
	model3Height = IniConfig::instance().getIniDouble("Model3", "height");
	model3Width = IniConfig::instance().getIniDouble("Model3", "width");
}

//void Model3Holder::createThreadPool(int threadNum)
//{
//	idlThrNum = threadNum;
//	totalThrNum = threadNum;
//	for (int size = 0; size < totalThrNum; ++size)
//	{   //初始化线程数量
//		pool.emplace_back(
//			[this]
//			{ // 工作线程函数
//				while (!this->stopped.load())
//				{
//					std::function<void()> task;
//					{   // 获取一个待执行的 task
//						std::unique_lock<std::mutex> lock{ this->task_mutex };// unique_lock 相比 lock_guard 的好处是：可以随时 unlock() 和 lock()
//						this->task_cv.wait(lock,
//							[this] {
//								return this->stopped.load() || !this->tasks.empty();
//							}
//						); // wait 直到有 task
//						if (this->stopped.load() && this->tasks.empty())
//							return;
//						task = std::move(this->tasks.front()); // 取一个 task
//						this->tasks.pop();
//					}
//					idlThrNum--;
//					task();
//					idlThrNum++;
//				}
//			}
//			);
//	}
//}

void Model3Holder::pushData(MultiImageRead& mImgRead)
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

//void Model3Holder::popQueueWithoutLock(vector<std::pair<cv::Rect, cv::Mat>>& rectMats)
//{
//	int size = data_queue.size();
//	for (int i = 0; i < size; i++)
//	{
//		rectMats.emplace_back(std::move(data_queue.front()));
//		data_queue.pop();
//	}
//}
//
//bool Model3Holder::popData(std::vector<std::pair<cv::Rect, cv::Mat>>& rectMats)
//{
//	std::unique_lock<std::mutex> data_lock(data_mutex);
//	if (data_queue.size() > 0)
//	{
//		popQueueWithoutLock(rectMats);
//		data_lock.unlock();
//		return true;
//	}
//	else
//	{
//		data_lock.unlock();
//		//取得tasks的锁，检查是否还有任务
//		std::unique_lock<std::mutex> task_lock(task_mutex);
//		if (tasks.size() > 0)
//		{
//			task_lock.unlock();
//			//证明有task，那么data_mutex再次锁上，因为一定会有数据唤醒它
//			data_lock.lock();
//			data_cv.wait(data_lock, [this] {
//				if (data_queue.size() > 0 || stopped.load()) {
//					return true;
//				}
//				else {
//					return false;
//				}
//				});
//			if (stopped.load())
//				return false;
//			popQueueWithoutLock(rectMats);
//			data_lock.unlock();
//			return true;
//		}
//		else
//		{
//			task_lock.unlock();
//			//如果没有task，那么要检查是否有线程在运行
//			if (idlThrNum.load() == totalThrNum.load())
//			{
//				//如果没有线程在运行，那么在看看队列是否有元素（万一在判断的时候进入了队列了呢？）
//				data_lock.lock();//其实不用加锁，因为没有线程在运行，肯定不会占用锁了
//				popQueueWithoutLock(rectMats);
//				data_lock.unlock();
//				if (rectMats.size() == 0)
//					return false;
//				return true;
//			}
//			else
//			{
//				//如果有线程在运行，那么在锁住队列，等待人来唤醒
//				data_lock.lock();
//				data_cv.wait_for(data_lock, 1000ms, [this] {
//					if (data_queue.size() > 0 || stopped.load()) {
//						return true;
//					}
//					else {
//						return false;
//					}
//					});
//				if (stopped.load())
//					return false;
//				//popQueueWithoutLock(rectMats);
//				data_lock.unlock();
//				return popData(rectMats);
//				//return true;
//			}
//		}
//	}
//}

cv::Point Model3Holder::rect2Point(int x, int y, float radius)
{
	cv::Point point(ceil(x + radius), ceil(y + radius));
	return point;
}

std::vector<PointScore> Model3Holder::model3Recom(std::vector<std::pair<cv::Rect, model3Result>>& xyResults)
{
	auto lambda = [](std::pair<cv::Rect, model3Result> result1, std::pair<cv::Rect, model3Result> result2)->bool {
		if (result1.second.type == result2.second.type)
		{
			if (result1.second.scores[result1.second.type] > result2.second.scores[result1.second.type])
				return true;
			else
				return false;
		}
		else if (result1.second.type < result2.second.type)
		{
			return true;
		}
		else
		{
			return false;
		}
	};
	std::sort(xyResults.begin(), xyResults.end(), lambda);
	std::vector<PointScore> retPs;
	//现有策略是只推荐出典型区域
	float radius = model3Width / 2 * float(model3Mpp / slideMpp);
	for (int i = 0; i < xyResults.size(); i++)
	{
		if (xyResults[i].second.type == model3Result::TYPICAL)
		{
			//retRects.emplace_back(xyResults[i].first);
			PointScore ps;
			ps.point = rect2Point(xyResults[i].first.x, xyResults[i].first.y, radius);
			ps.score = xyResults[i].second.scores[0];
			retPs.emplace_back(ps);
		}
	}
	return retPs;
}

std::vector<PointScore> Model3Holder::runModel3(MultiImageRead& mImgRead, std::vector<Anno> &annos)
{
	initPara(mImgRead);
	mImgRead.setGammaFlag(false);
	std::vector<cv::Rect> rects;
	for (int i = 0; i < annos.size(); i++) {
		cv::Rect rect;
		rect.x = annos[i].x - model3Width / 2 * float(model3Mpp / slideMpp);
		rect.y = annos[i].y - model3Height / 2 * float(model3Mpp / slideMpp);
		rect.height = model3Height * float(model3Mpp / slideMpp);
		rect.width = model3Width * float(model3Mpp / slideMpp);
		rects.emplace_back(rect);
	}
	mImgRead.setRects(rects);
	
	std::vector<std::pair<cv::Rect, cv::Mat>> rectMats;
	std::vector<std::pair<cv::Rect, cv::Mat>> tmpRectMats;

	for (int i = 0; i < totalThrNum; i++)
	{
		auto task = std::make_shared<std::packaged_task<void()>>
			(std::bind(&Model3Holder::pushData, this, std::ref(mImgRead)));
		std::unique_lock<std::mutex> task_lock(task_mutex);
		tasks.emplace(
			[task]() {
				(*task)();
			}
		);
		task_lock.unlock();
		task_cv.notify_one();
	}
	while (popData(tmpRectMats)) {
		for (auto iter = tmpRectMats.begin(); iter != tmpRectMats.end(); iter++) {
			rectMats.emplace_back(std::move(*iter));
		}
		tmpRectMats.clear();
	}

	std::vector<cv::Mat> imgs;
	std::vector<std::pair<cv::Rect, model3Result>> xyResults;
	for (auto& elem : rectMats)
	{
		imgs.emplace_back(std::move(elem.second));
	}
	//开始预测50张图像
	model3Handle->processDataConcurrency(imgs);
	std::vector<model3Result> results = model3Handle->m_results;
	for (auto& elem : results)
	{
		elem.iniType();
	}
	for (auto iter = results.begin(); iter != results.end(); iter++)
	{
		int place = iter - results.begin();
		std::pair<cv::Rect, model3Result> xyResult;
		xyResult.first = rectMats[place].first;
		xyResult.second = *iter;
		xyResults.emplace_back(xyResult);
	}
	//返回model3
	return model3Recom(xyResults);
}