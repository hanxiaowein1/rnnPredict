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
	std::unique_ptr<SlideFactory> sFactory(new SlideFactory());
	for (int i = 0; i < 8; i++)
	{
		sReads.emplace_back(sFactory->createSlideProduct(slidePath));
	}
	stopped.store(false);
	std::vector<std::mutex> list(sReads.size());
	lock_sRead.swap(list);
}

MultiImageRead::~MultiImageRead()
{
	stopped.store(true);
	cv_task.notify_all();
	cv_rects.notify_all();
	cv_queue_has_elem.notify_all();
	cv_queue_overflow.notify_all();
	while (true)
	{
		//Sleep(3000);	
		if (!addTaskFlag.load())
			break;
		cout << "卡在addTaskFlag不为false上了\n";
		Sleep(500);
	}
	for (std::thread& thread : pool) {
		if (thread.joinable())
			thread.join();
	}
	cout << "所有线程已经结束\n";
}



void MultiImageRead::createThreadPool()
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

//往里面添加task，为addTask单独设置线程
void MultiImageRead::addTask()
{
	//循环0 - sReads的长度，往里面加入task
	while (!stopped.load())
	{
		for (int i = 0; i < sReads.size(); i++)
		{			
			std::unique_lock<std::mutex> myGuard2(rectMutex);
			if (m_rects.size() == 0)
			{
				cv_rects.wait(myGuard2, [this] {
					if (m_rects.size() > 0 || stopped.load())
						return true;
					else
						return false;
					});
			}
			if (stopped.load())
			{
				addTaskFlag.store(false);
				return;
			}
			cv::Rect rect = std::move(m_rects.front());
			m_rects.pop();
			myGuard2.unlock();
			std::unique_lock<std::mutex> lock(m_lock);
			//std::cout << "before enter task" << std::endl;
			auto task = std::make_shared<std::packaged_task<void()>>(std::bind(&MultiImageRead::readTask, this, i, rect));
			TaskCount++;
			tasks.emplace(
				[task]() {
					(*task)();
				}
			);	
			lock.unlock();
			//std::cout << "after enter task" << endl;
			cv_task.notify_one();
		}
	}
	
}

void MultiImageRead::setAddTaskThread()
{
	std::thread tempThread(&MultiImageRead::addTask, this);
	tempThread.detach();
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


void MultiImageRead::readTask(int i, cv::Rect &rect)
{	
	//checkPoint1++;
	//独占SlideRead
	std::unique_lock<std::mutex> myGuard(lock_sRead[i]);
	std::unique_ptr<SlideRead>& uptr = sReads[i];
	//checkPoint2++;
	std::pair<cv::Rect, cv::Mat> rectMat;
	uptr->getTile(read_level, rect.x, rect.y, rect.width, rect.height, rectMat.second);
	//经检查，checkPoint2到checkPoint3之间有错误，存在几个函数没有执行完毕
	//checkPoint3++;
	rectMat.first = std::move(rect);
	//在这里对齐进行gamma变换
	if (gamma_flag.load())
	{
		m_GammaCorrection(rectMat.second, rectMat.second, 0.6f);
	}	
	//为pair队列加锁
	std::unique_lock<std::mutex> myGuard3(mutex_mat);
	//checkPoint4++;
	//如果pair队列长度大于200，那么就等一会在执行
	if (m_rmQueue.size() >= maxQueueNum)
	{
		cv_queue_overflow.wait(myGuard3, [this] {
			if (m_rmQueue.size() < maxQueueNum || stopped.load()) {
				return true;
			}
			else {
				return false;
			}
		});
		if (stopped.load())
			return;
		m_rmQueue.emplace(std::move(rectMat));
		myGuard3.unlock();
		cv_queue_has_elem.notify_one();//已经有元素了，通知pop
	}
	else
	{
		//checkPoint5++;
		m_rmQueue.emplace(std::move(rectMat));
		myGuard3.unlock();
		cv_queue_has_elem.notify_one();//已经有元素了，通知pop
	}
	readTaskCount++;
	if (readTaskCount.load() == totalTaskCount)
	{
		time_t now = time(0);
		cout << "read image to buffer complete: " << (char*)ctime(&now);
	}
}

bool MultiImageRead::popQueue(std::vector<std::pair<cv::Rect, cv::Mat>>& rectMats)
{
	//更改策略，每次pop出1/4*(maxQueueNum)的元素，以供外部线程平均处理，(如果队列里有数据，就全部pop出，反正也是std::move()，无需拷贝时间很快...)
	std::unique_lock<std::mutex> myGuard3(mutex_mat);
	if (m_rmQueue.size() > 0)
	{
		int size = m_rmQueue.size();
		int popNum = 0;
		if (size > (maxQueueNum / 4))
			popNum = maxQueueNum / 4;
		else
			popNum = size;
		//auto start = std::chrono::system_clock::now();
		for (int i = 0; i < popNum; i++)
		{
			rectMats.emplace_back(std::move(m_rmQueue.front()));
			m_rmQueue.pop();
		}
		/*auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end - start;
		std::cout << "pop "<<size<<" elements time is "
			<< diff.count() << " s\n";*/
		myGuard3.unlock();
		cv_queue_overflow.notify_one();
		return true;
	}
	else
	{
		//如果没有数据，就先释放锁
		//在判断是否有tasks
		std::unique_lock<std::mutex> lock{ this->m_lock };
		//当有task或者空闲的线程数量也不等于全部的线程数量(也就是说还有线程正在执行)
		if (tasks.size() > 0)
		{
			//有tasks，释放锁
			lock.unlock();
			//等待有元素
			cv_queue_has_elem.wait(myGuard3, [this] {
				if (m_rmQueue.size() > 0 || stopped.load()) {
					return true;
				}
				else {
					return false;
				}
				});
			if (stopped.load())
				return false;
			//弹出m_rmQueue
			rectMats.emplace_back(std::move(m_rmQueue.front()));
			m_rmQueue.pop();
			myGuard3.unlock();
			cv_queue_overflow.notify_one();//已经pop出一个元素，通知进图
			return true;
		}	
		if (tasks.size() == 0 && idlThrNum != totalThrNum)
		{
			//证明有线程正在运行
			lock.unlock();
			cv_queue_has_elem.wait_for(myGuard3, 5000ms, [this] {
				if (m_rmQueue.size() > 0 || stopped.load()) {
					return true;
				}
				else {
					return false;
				}
				});
			if (stopped.load())
				return false;
			if (m_rmQueue.size() ==0 && (idlThrNum != totalThrNum))
			{
				//线程等了5秒还在运行
				myGuard3.unlock();
				bool flag = popQueue(rectMats);
				return flag;
			}
			if (m_rmQueue.size() == 0 && idlThrNum == totalThrNum)
			{
				return false;
			}
			//弹出m_rmQueue
			rectMats.emplace_back(std::move(m_rmQueue.front()));
			m_rmQueue.pop();
			myGuard3.unlock();
			cv_queue_overflow.notify_one();//已经pop出一个元素，通知进图
			return true;
		}
		//然后判断m_rects是否有值(可能正处在addTask往tasks里加任务的时候)
		if (m_rects.size() > 0 && addTaskFlag.load())
		{
			lock.unlock();
			//再次等待添加queue的线程通知
			cv_queue_has_elem.wait(myGuard3, [this] {
				if (m_rmQueue.size() > 0 || stopped.load()) {
					return true;
				}
				else {
					return false;
				}
				});
			if (stopped.load())
				return false;
			//弹出m_rmQueue
			rectMats.emplace_back(std::move(m_rmQueue.front()));
			m_rmQueue.pop();
			myGuard3.unlock();
			cv_queue_overflow.notify_one();
			return true;

		}
		return false;
	}
}

bool MultiImageRead::popQueue(std::pair<cv::Rect, cv::Mat>& rectMat)
{
	//首先判断队列里是否有数据
	std::unique_lock<std::mutex> myGuard3(mutex_mat);
	if (m_rmQueue.size() > 0)
	{
		rectMat = std::move(m_rmQueue.front());
		m_rmQueue.pop();
		myGuard3.unlock();
		cv_queue_overflow.notify_one();
		return true;
	}
	else
	{
		//如果没有数据，先释放锁
		//在判断是否有tasks
		std::unique_lock<std::mutex> lock{ this->m_lock };
		if (tasks.size() > 0)
		{
			//有tasks，释放锁
			lock.unlock();
			//等待有元素
			cv_queue_has_elem.wait(myGuard3, [this] {
				if (m_rmQueue.size() > 0 || stopped.load()) {
					return true;
				}
				else {
					return false;
				}
			});
			if (stopped.load())
				return false;
			//弹出m_rmQueue
			rectMat = std::move(m_rmQueue.front());
			m_rmQueue.pop();
			myGuard3.unlock();
			cv_queue_overflow.notify_one();//已经pop出一个元素，通知进图
			return true;
		}
		if (tasks.size() == 0 && idlThrNum != totalThrNum)
		{
			//证明有线程正在运行
			lock.unlock();
			cv_queue_has_elem.wait(myGuard3, [this] {
				if (m_rmQueue.size() > 0 || stopped.load()) {
					return true;
				}
				else {
					return false;
				}
				});
			if (stopped.load())
				return false;
			if (m_rmQueue.size() == 0 && (idlThrNum != totalThrNum))
			{
				//线程等了5秒还在运行
				myGuard3.unlock();
				bool flag = popQueue(rectMat);
				return flag;
			}
			if (m_rmQueue.size() == 0 && idlThrNum == totalThrNum)
			{
				return false;
			}
			//弹出m_rmQueue
			rectMat = std::move(m_rmQueue.front());
			m_rmQueue.pop();
			myGuard3.unlock();
			cv_queue_overflow.notify_one();//已经pop出一个元素，通知进图
			return true;
		}
		//然后判断m_rects是否有值(可能正处在addTask往tasks里加任务的时候)
		if (m_rects.size() > 0 && addTaskFlag.load())
		{
			//如果没有tasks了，还是先释放tasks锁
			lock.unlock();
			//再次等待添加queue的线程通知
			cv_queue_has_elem.wait(myGuard3, [this] {
				if (m_rmQueue.size() > 0 || stopped.load()) {
					return true;
				}
				else {
					return false;
				}
				});
			if (stopped.load())
				return false;
			//弹出m_rmQueue
			rectMat = std::move(m_rmQueue.front());
			m_rmQueue.pop();
			myGuard3.unlock();
			cv_queue_overflow.notify_one();
			return true;
		}
		return false;	
	}	
}

void MultiImageRead::setRects(std::vector<cv::Rect>& rects)
{
	if (rects.size() == 0)
		return;
	std::unique_lock<std::mutex> myGuard2(rectMutex);
	//先将老的缓存清理掉
	//clear(m_rects);
	for (int i = 0; i < rects.size(); i++)
	{
		m_rects.emplace(rects[i]);
	}
	int size = rects.size();
	totalTaskCount += size;
	myGuard2.unlock();
	cv_rects.notify_one();
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





