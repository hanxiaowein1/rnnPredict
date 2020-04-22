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
		cout << "����addTaskFlag��Ϊfalse����\n";
		Sleep(500);
	}
	for (std::thread& thread : pool) {
		if (thread.joinable())
			thread.join();
	}
	cout << "�����߳��Ѿ�����\n";
}



void MultiImageRead::createThreadPool()
{
	int num = idlThrNum;
	for (int size = 0; size < num; ++size)
	{   //��ʼ���߳�����
		pool.emplace_back(
			[this]
			{ // �����̺߳���
				while (!this->stopped.load())
				{
					std::function<void()> task;
					{   // ��ȡһ����ִ�е� task
						std::unique_lock<std::mutex> lock{ this->m_lock };// unique_lock ��� lock_guard �ĺô��ǣ�������ʱ unlock() �� lock()
						this->cv_task.wait(lock,
							[this] {
								return this->stopped.load() || !this->tasks.empty();
							}
						); // wait ֱ���� task
						if (this->stopped.load() && this->tasks.empty())
							return;
						task = std::move(this->tasks.front()); // ȡһ�� task
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

//���������task��ΪaddTask���������߳�
void MultiImageRead::addTask()
{
	//ѭ��0 - sReads�ĳ��ȣ����������task
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
	//��ռSlideRead
	std::unique_lock<std::mutex> myGuard(lock_sRead[i]);
	std::unique_ptr<SlideRead>& uptr = sReads[i];
	//checkPoint2++;
	std::pair<cv::Rect, cv::Mat> rectMat;
	uptr->getTile(read_level, rect.x, rect.y, rect.width, rect.height, rectMat.second);
	//����飬checkPoint2��checkPoint3֮���д��󣬴��ڼ�������û��ִ�����
	//checkPoint3++;
	rectMat.first = std::move(rect);
	//������������gamma�任
	if (gamma_flag.load())
	{
		m_GammaCorrection(rectMat.second, rectMat.second, 0.6f);
	}	
	//Ϊpair���м���
	std::unique_lock<std::mutex> myGuard3(mutex_mat);
	//checkPoint4++;
	//���pair���г��ȴ���200����ô�͵�һ����ִ��
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
		cv_queue_has_elem.notify_one();//�Ѿ���Ԫ���ˣ�֪ͨpop
	}
	else
	{
		//checkPoint5++;
		m_rmQueue.emplace(std::move(rectMat));
		myGuard3.unlock();
		cv_queue_has_elem.notify_one();//�Ѿ���Ԫ���ˣ�֪ͨpop
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
	//���Ĳ��ԣ�ÿ��pop��1/4*(maxQueueNum)��Ԫ�أ��Թ��ⲿ�߳�ƽ������(��������������ݣ���ȫ��pop��������Ҳ��std::move()�����追��ʱ��ܿ�...)
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
		//���û�����ݣ������ͷ���
		//���ж��Ƿ���tasks
		std::unique_lock<std::mutex> lock{ this->m_lock };
		//����task���߿��е��߳�����Ҳ������ȫ�����߳�����(Ҳ����˵�����߳�����ִ��)
		if (tasks.size() > 0)
		{
			//��tasks���ͷ���
			lock.unlock();
			//�ȴ���Ԫ��
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
			//����m_rmQueue
			rectMats.emplace_back(std::move(m_rmQueue.front()));
			m_rmQueue.pop();
			myGuard3.unlock();
			cv_queue_overflow.notify_one();//�Ѿ�pop��һ��Ԫ�أ�֪ͨ��ͼ
			return true;
		}	
		if (tasks.size() == 0 && idlThrNum != totalThrNum)
		{
			//֤�����߳���������
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
				//�̵߳���5�뻹������
				myGuard3.unlock();
				bool flag = popQueue(rectMats);
				return flag;
			}
			if (m_rmQueue.size() == 0 && idlThrNum == totalThrNum)
			{
				return false;
			}
			//����m_rmQueue
			rectMats.emplace_back(std::move(m_rmQueue.front()));
			m_rmQueue.pop();
			myGuard3.unlock();
			cv_queue_overflow.notify_one();//�Ѿ�pop��һ��Ԫ�أ�֪ͨ��ͼ
			return true;
		}
		//Ȼ���ж�m_rects�Ƿ���ֵ(����������addTask��tasks��������ʱ��)
		if (m_rects.size() > 0 && addTaskFlag.load())
		{
			lock.unlock();
			//�ٴεȴ����queue���߳�֪ͨ
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
			//����m_rmQueue
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
	//�����ж϶������Ƿ�������
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
		//���û�����ݣ����ͷ���
		//���ж��Ƿ���tasks
		std::unique_lock<std::mutex> lock{ this->m_lock };
		if (tasks.size() > 0)
		{
			//��tasks���ͷ���
			lock.unlock();
			//�ȴ���Ԫ��
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
			//����m_rmQueue
			rectMat = std::move(m_rmQueue.front());
			m_rmQueue.pop();
			myGuard3.unlock();
			cv_queue_overflow.notify_one();//�Ѿ�pop��һ��Ԫ�أ�֪ͨ��ͼ
			return true;
		}
		if (tasks.size() == 0 && idlThrNum != totalThrNum)
		{
			//֤�����߳���������
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
				//�̵߳���5�뻹������
				myGuard3.unlock();
				bool flag = popQueue(rectMat);
				return flag;
			}
			if (m_rmQueue.size() == 0 && idlThrNum == totalThrNum)
			{
				return false;
			}
			//����m_rmQueue
			rectMat = std::move(m_rmQueue.front());
			m_rmQueue.pop();
			myGuard3.unlock();
			cv_queue_overflow.notify_one();//�Ѿ�pop��һ��Ԫ�أ�֪ͨ��ͼ
			return true;
		}
		//Ȼ���ж�m_rects�Ƿ���ֵ(����������addTask��tasks��������ʱ��)
		if (m_rects.size() > 0 && addTaskFlag.load())
		{
			//���û��tasks�ˣ��������ͷ�tasks��
			lock.unlock();
			//�ٴεȴ����queue���߳�֪ͨ
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
			//����m_rmQueue
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
	//�Ƚ��ϵĻ��������
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





