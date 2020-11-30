#include "MultiImageRead.h"

#include <chrono>

void clear(std::queue<cv::Rect>& q)
{
	std::queue<cv::Rect> empty;
	std::swap(q, empty);
}
void clear(std::queue<std::pair<cv::Rect, cv::Mat>>& q)
{
	std::queue<std::pair<cv::Rect, cv::Mat>> empty;
	std::swap(q, empty);
}

MultiImageRead::MultiImageRead(const char* slidePath)
{
	m_slidePath = std::string(slidePath);

	stopped.store(false);
}

MultiImageRead::~MultiImageRead()
{
	stopped.store(true);
	task_cv.notify_all();
	cv_queue_has_elem.notify_all();
	cv_queue_overflow.notify_all();
	for (std::thread& thread : pool) {
		if (thread.joinable())
			thread.join();
	}
	std::cout << "所有线程已经结束\n";
}



void MultiImageRead::createThreadPool()
{
	std::unique_ptr<SlideFactory> sFactory(new SlideFactory());
	for (int i = 0; i < totalThrNum; i++)
	{
		sReads.emplace_back(sFactory->createSlideProduct(m_slidePath.c_str()));
	}
	std::vector<std::mutex> list(sReads.size());
	sRead_mutex.swap(list);
	int num = totalThrNum;

	for (int size = 0; size < num; ++size)
	{   //初始化线程数量
		pool.emplace_back(
			[this]
			{ // 工作线程函数
				while (!this->stopped.load())
				{
					std::function<void()> task;
					{   // 获取一个待执行的 task
						std::unique_lock<std::mutex> task_lock{ this->task_mutex };// unique_lock 相比 lock_guard 的好处是：可以随时 unlock() 和 lock()
						this->task_cv.wait(task_lock,
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

void MultiImageRead::m_GammaCorrection(cv::Mat& src, cv::Mat& dst, float fGamma)
{
	unsigned char lut[256];
	for (int i = 0; i < 256; i++) {
		lut[i] = cv::saturate_cast<uchar>(int(pow((float)(i / 255.0), fGamma) * 255.0f));
	}
	//dst = src.clone();
	const int channels = dst.channels();
	switch (channels) {
	case 1: {
		cv::MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
			*it = lut[(*it)];
		break;
	}
	case 3: {
		for (int i = 0; i < dst.rows; i++) {
			uchar* linePtr = dst.ptr(i);
			for (int j = 0; j < dst.cols; j++) {
				*(linePtr + j * 3) = lut[*(linePtr + j * 3)];
				*(linePtr + j * 3 + 1) = lut[*(linePtr + j * 3 + 1)];
				*(linePtr + j * 3 + 2) = lut[*(linePtr + j * 3 + 2)];
			}
		}
		break;
	}
	}
}

void MultiImageRead::readTask(int i, cv::Rect rect)
{	
	//独占SlideRead
	std::unique_lock<std::mutex> sRead_lock(sRead_mutex[i]);
	std::unique_ptr<SlideRead>& uptr = sReads[i];
	std::pair<cv::Rect, cv::Mat> rectMat;
	uptr->getTile(read_level, rect.x, rect.y, rect.width, rect.height, rectMat.second);
	//经检查，checkPoint2到checkPoint3之间有错误，存在几个函数没有执行完毕
	rectMat.first = rect;
	//在这里对齐进行gamma变换
	if (gamma_flag.load())
	{
		m_GammaCorrection(rectMat.second, rectMat.second, 0.6f);
	}	
	//为pair队列加锁
	std::unique_lock<std::mutex> data_lock(data_mutex);
	//如果pair队列长度大于200，那么就等一会在执行
	if (data_queue.size() >= maxQueueNum.load())
	{
		cv_queue_overflow.wait(data_lock, [this] {
			if (data_queue.size() < maxQueueNum.load() || stopped.load()) {
				return true;
			}
			else {
				return false;
			}
		});
		if (stopped.load())
			return;
		data_queue.emplace(std::move(rectMat));
		data_lock.unlock();
		cv_queue_has_elem.notify_one();//已经有元素了，通知pop
	}
	else
	{
		data_queue.emplace(std::move(rectMat));
		data_lock.unlock();
		cv_queue_has_elem.notify_one();//已经有元素了，通知pop
	}
	readTaskCount++;
	if (readTaskCount.load() == totalTaskCount.load())
	{
		time_t now = time(0);
		std::cout << "read image to buffer complete: " << (char*)ctime(&now);
	}
}

void MultiImageRead::popQueueWithoutLock(std::vector<std::pair<cv::Rect, cv::Mat>>& rectMats)
{
	int size = data_queue.size();
	for (int i = 0; i < size; i++)
	{
		rectMats.emplace_back(std::move(data_queue.front()));
		data_queue.pop();
	}
}

bool MultiImageRead::popQueue(std::vector<std::pair<cv::Rect, cv::Mat>>& rectMats)
{
	//如果队列里有数据，就全部pop出，反正也是std::move()，无需拷贝时间很快
	std::unique_lock<std::mutex> data_lock(data_mutex);
	if (data_queue.size() > 0)
	{
		popQueueWithoutLock(rectMats);
		data_lock.unlock();
		cv_queue_overflow.notify_one();
		return true;
	}
	else
	{
		//如果没有数据，就先释放锁
		data_lock.unlock();
		//在判断是否有tasks
		std::unique_lock<std::mutex> task_lock{ task_mutex };
		//当有task或者空闲的线程数量也不等于全部的线程数量(也就是说还有线程正在执行)
		if (tasks.size() > 0)
		{
			//有tasks，释放锁
			task_lock.unlock();
			//等待有元素
			data_lock.lock();
			cv_queue_has_elem.wait(data_lock, [this] {
				if (data_queue.size() > 0 || stopped.load()) {
					return true;
				}
				else {
					return false;
				}
				});
			if (stopped.load())
				return false;
			//弹出m_rmQueue
			popQueueWithoutLock(rectMats);
			data_lock.unlock();
			cv_queue_overflow.notify_all();//已经pop元素，通知进图
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
				cv_queue_overflow.notify_all();//已经pop元素，通知进图
				if (rectMats.size() == 0)
					return false;
				return true;
			}
			else
			{
				//如果有线程在运行，那么在锁住队列，等待人来唤醒
				data_lock.lock();
				using namespace std;
				cv_queue_has_elem.wait_for(data_lock, 1000ms, [this] {
					if (data_queue.size() > 0 || stopped.load()) {
						return true;
					}
					else {
						return false;
					}
					});
				if (stopped.load())
					return false;
				//popQueueWithoutLock(rectMats);
				data_lock.unlock();
				return popQueue(rectMats);
			}
		}
	}
}

bool MultiImageRead::popQueue(std::pair<cv::Rect, cv::Mat>& rectMat)
{
	//首先判断队列里是否有数据
	std::unique_lock<std::mutex> data_lock(data_mutex);
	if (data_queue.size() > 0)
	{
		rectMat = std::move(data_queue.front());
		data_queue.pop();
		data_lock.unlock();
		cv_queue_overflow.notify_one();
		return true;
	}
	else
	{
		//如果没有数据，先释放锁
		//在判断是否有tasks
		std::unique_lock<std::mutex> task_lock{ this->task_mutex };
		if (tasks.size() > 0)
		{
			//有tasks，释放锁
			task_lock.unlock();
			//等待有元素
			cv_queue_has_elem.wait(data_lock, [this] {
				if (data_queue.size() > 0 || stopped.load()) {
					return true;
				}
				else {
					return false;
				}
			});
			if (stopped.load())
				return false;
			//弹出m_rmQueue
			rectMat = std::move(data_queue.front());
			data_queue.pop();
			data_lock.unlock();
			cv_queue_overflow.notify_one();//已经pop出一个元素，通知进图
			return true;
		}
		if (tasks.size() == 0 && idlThrNum != totalThrNum)
		{
			//证明有线程正在运行
			task_lock.unlock();
			cv_queue_has_elem.wait(data_lock, [this] {
				if (data_queue.size() > 0 || stopped.load()) {
					return true;
				}
				else {
					return false;
				}
				});
			if (stopped.load())
				return false;
			if (data_queue.size() == 0 && (idlThrNum != totalThrNum))
			{
				//线程等了5秒还在运行
				data_lock.unlock();
				bool flag = popQueue(rectMat);
				return flag;
			}
			if (data_queue.size() == 0 && idlThrNum == totalThrNum)
			{
				return false;
			}
			//弹出m_rmQueue
			rectMat = std::move(data_queue.front());
			data_queue.pop();
			data_lock.unlock();
			cv_queue_overflow.notify_one();//已经pop出一个元素，通知进图
			return true;
		}
		return false;	
	}	
}

void MultiImageRead::setRects(std::vector<cv::Rect> rects)
{
	//TaskCount = 0;
	if (rects.size() == 0)
		return;
	for (int i = 0; i < rects.size(); i++)
	{
		cv::Rect rect = rects[i];
		std::unique_lock<std::mutex> task_lock(task_mutex);
		auto task = std::make_shared<std::packaged_task<void()>>(std::bind(&MultiImageRead::readTask, this, i % totalThrNum, rect));
		totalTaskCount++;
		tasks.emplace(
			[task]() {
				(*task)();
			}
		);
		task_lock.unlock();
	}
	task_cv.notify_all();
}

bool MultiImageRead::status()
{
	if (sReads.size() > 0)
	{
		if (sReads[0]->status())
			return true;
		return false;
	}
	return false;
}

void MultiImageRead::getSlideWidth(int& width)
{
	if (sReads.size() > 0)
	{
		sReads[0]->getSlideWidth(width);
	}
}

void MultiImageRead::getSlideHeight(int& height)
{
	if (sReads.size() > 0)
	{
		sReads[0]->getSlideHeight(height);
	}
}

void MultiImageRead::getSlideBoundX(int& boundX)
{
	if (sReads.size() > 0)
	{
		sReads[0]->getSlideBoundX(boundX);
	}
}

void MultiImageRead::getSlideBoundY(int& boundY)
{
	if (sReads.size() > 0)
	{
		sReads[0]->getSlideBoundY(boundY);
	}
}

void MultiImageRead::getSlideMpp(double& mpp)
{
	if (sReads.size() > 0)
	{
		sReads[0]->getSlideMpp(mpp);
	}
}

void MultiImageRead::getLevelDimensions(int level, int& width, int& height)
{
	if (sReads.size() > 0)
	{
		sReads[0]->getLevelDimensions(level, width, height);
	}
}

void MultiImageRead::getTile(int level, int x, int y, int width, int height, cv::Mat& img)
{
	if (sReads.size() > 0)
	{
		sReads[0]->getTile(level, x, y, width, height, img);
	}
}

int MultiImageRead::get_ratio()
{
	if (sReads.size() > 0)
	{
		sReads[0]->ini_ration();
		return sReads[0]->m_ratio;
	}
	return -1;
}

std::unique_ptr<SlideRead> MultiImageRead::getSingleReadHandleAndReleaseOthers()
{
	if (sReads.size() > 0)
	{
		for (int i = 1; i < sReads.size(); i++)
		{
			sReads[i].reset(nullptr);
		}
	}
	std::unique_ptr<SlideRead> ret_ptr = std::move(sReads[0]);
	return ret_ptr;
}



