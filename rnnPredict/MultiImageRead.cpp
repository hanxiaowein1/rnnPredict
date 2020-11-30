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
	std::cout << "�����߳��Ѿ�����\n";
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
	{   //��ʼ���߳�����
		pool.emplace_back(
			[this]
			{ // �����̺߳���
				while (!this->stopped.load())
				{
					std::function<void()> task;
					{   // ��ȡһ����ִ�е� task
						std::unique_lock<std::mutex> task_lock{ this->task_mutex };// unique_lock ��� lock_guard �ĺô��ǣ�������ʱ unlock() �� lock()
						this->task_cv.wait(task_lock,
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
	//��ռSlideRead
	std::unique_lock<std::mutex> sRead_lock(sRead_mutex[i]);
	std::unique_ptr<SlideRead>& uptr = sReads[i];
	std::pair<cv::Rect, cv::Mat> rectMat;
	uptr->getTile(read_level, rect.x, rect.y, rect.width, rect.height, rectMat.second);
	//����飬checkPoint2��checkPoint3֮���д��󣬴��ڼ�������û��ִ�����
	rectMat.first = rect;
	//������������gamma�任
	if (gamma_flag.load())
	{
		m_GammaCorrection(rectMat.second, rectMat.second, 0.6f);
	}	
	//Ϊpair���м���
	std::unique_lock<std::mutex> data_lock(data_mutex);
	//���pair���г��ȴ���200����ô�͵�һ����ִ��
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
		cv_queue_has_elem.notify_one();//�Ѿ���Ԫ���ˣ�֪ͨpop
	}
	else
	{
		data_queue.emplace(std::move(rectMat));
		data_lock.unlock();
		cv_queue_has_elem.notify_one();//�Ѿ���Ԫ���ˣ�֪ͨpop
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
	//��������������ݣ���ȫ��pop��������Ҳ��std::move()�����追��ʱ��ܿ�
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
		//���û�����ݣ������ͷ���
		data_lock.unlock();
		//���ж��Ƿ���tasks
		std::unique_lock<std::mutex> task_lock{ task_mutex };
		//����task���߿��е��߳�����Ҳ������ȫ�����߳�����(Ҳ����˵�����߳�����ִ��)
		if (tasks.size() > 0)
		{
			//��tasks���ͷ���
			task_lock.unlock();
			//�ȴ���Ԫ��
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
			//����m_rmQueue
			popQueueWithoutLock(rectMats);
			data_lock.unlock();
			cv_queue_overflow.notify_all();//�Ѿ�popԪ�أ�֪ͨ��ͼ
			return true;
		}	
		else
		{
			task_lock.unlock();
			//���û��task����ôҪ����Ƿ����߳�������
			if (idlThrNum.load() == totalThrNum.load())
			{
				//���û���߳������У���ô�ڿ��������Ƿ���Ԫ�أ���һ���жϵ�ʱ������˶������أ���
				data_lock.lock();//��ʵ���ü�������Ϊû���߳������У��϶�����ռ������
				popQueueWithoutLock(rectMats);
				data_lock.unlock();
				cv_queue_overflow.notify_all();//�Ѿ�popԪ�أ�֪ͨ��ͼ
				if (rectMats.size() == 0)
					return false;
				return true;
			}
			else
			{
				//������߳������У���ô����ס���У��ȴ���������
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
	//�����ж϶������Ƿ�������
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
		//���û�����ݣ����ͷ���
		//���ж��Ƿ���tasks
		std::unique_lock<std::mutex> task_lock{ this->task_mutex };
		if (tasks.size() > 0)
		{
			//��tasks���ͷ���
			task_lock.unlock();
			//�ȴ���Ԫ��
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
			//����m_rmQueue
			rectMat = std::move(data_queue.front());
			data_queue.pop();
			data_lock.unlock();
			cv_queue_overflow.notify_one();//�Ѿ�pop��һ��Ԫ�أ�֪ͨ��ͼ
			return true;
		}
		if (tasks.size() == 0 && idlThrNum != totalThrNum)
		{
			//֤�����߳���������
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
				//�̵߳���5�뻹������
				data_lock.unlock();
				bool flag = popQueue(rectMat);
				return flag;
			}
			if (data_queue.size() == 0 && idlThrNum == totalThrNum)
			{
				return false;
			}
			//����m_rmQueue
			rectMat = std::move(data_queue.front());
			data_queue.pop();
			data_lock.unlock();
			cv_queue_overflow.notify_one();//�Ѿ�pop��һ��Ԫ�أ�֪ͨ��ͼ
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



