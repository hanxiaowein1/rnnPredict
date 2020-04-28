#include "Model2Holder.h"

Model2Holder::Model2Holder()
{
}

Model2Holder::Model2Holder(std::string iniPath)
{
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
	delete model2Handle;
}

void Model2Holder::initPara(MultiImageRead& mImgRead)
{
	mImgRead.getSlideHeight(slideHeight);
	mImgRead.getSlideWidth(slideWidth);
	mImgRead.getSlideMpp(slideMpp);

	model1Height = 512;
	model1Width = 512;
	model1Mpp = 0.586f;
}

void Model2Holder::model2Config(std::string iniPath)
{
	model2Handle = new TrModel2(iniPath, "TrModel2");
	model2Handle->createThreadPool();
	model2Mpp = model2Handle->inputProp.mpp;
	model2Height = model2Handle->inputProp.height;
	model2Width = model2Handle->inputProp.width;
}

void Model2Holder::createThreadPool(int threadNum)
{
	idlThrNum = threadNum;
	totalThrNum = threadNum;
	for (int size = 0; size < totalThrNum; ++size)
	{   //��ʼ���߳�����
		pool.emplace_back(
			[this]
			{ // �����̺߳���
				while (!this->stopped.load())
				{
					std::function<void()> task;
					{   // ��ȡһ����ִ�е� task
						std::unique_lock<std::mutex> lock{ this->task_mutex };// unique_lock ��� lock_guard �ĺô��ǣ�������ʱ unlock() �� lock()
						this->task_cv.wait(lock,
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
		//ȡ��tasks����������Ƿ�������
		std::unique_lock<std::mutex> task_lock(task_mutex);
		if (tasks.size() > 0)
		{
			task_lock.unlock();
			//֤����task����ôdata_mutex�ٴ����ϣ���Ϊһ���������ݻ�����
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
			//���û��task����ôҪ����Ƿ����߳�������
			if (idlThrNum.load() == totalThrNum.load())
			{
				//���û���߳������У���ô�ڿ��������Ƿ���Ԫ�أ���һ���жϵ�ʱ������˶������أ���
				data_lock.lock();//��ʵ���ü�������Ϊû���߳������У��϶�����ռ������
				popQueueWithoutLock(rectMats);
				data_lock.unlock();
				if (rectMats.size() == 0)
					return false;
				return true;
			}
			else
			{
				//������߳������У���ô����ס���У��ȴ���������
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
	model2Handle->processDataConcurrency(imgs);
	results = model2Handle->m_results;
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
	//һ���ص�Ĵ��󣡣�rnn�����tensorӦ���ǴӴ�С��
	//����flags����Ϊȥ��֮�����겻������ͬ
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
	//��rResults������ѡ����model2�Ŀ�
	//���շ����Ӵ�С����
	int m2MinNum = 600;
	int m2MaxNum = 1200;

	sortResultsByScore(rResults);
	std::vector<cv::Rect> rects;
	int placeStop = 0;

	//�ҿ���һ��ʼ�ͽ�model1��λ��Խ���ȫ���ɵ���...
	//��copyһ�����������rResults��Ȼ���ٴӸ�������������ķŵ�rResults����
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
		//�Ƚ�����0.5��ȫ���͵�����
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
				//����������
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
	//placeStop����0Ϊ���ģ����Լ���ʱ��Ҫ�����һ
	placeStop++;
	mImgRead.setReadLevel(0);//���㲻�䣬model2һֱ���Ǵ�level0�Ͻ��ж�ȡ
	mImgRead.setQueueMaxNum(rects.size());
	startRead(rects, mImgRead);

	std::vector<std::pair<cv::Rect, cv::Mat>> rectMats;
	std::vector<PointScore> model2PS;
	while (popData(rectMats))
	{
		std::vector<cv::Mat> imgs;
		std::vector<cv::Rect> tmpRects;
		for (auto iter = rectMats.begin(); iter != rectMats.end(); iter++) {
			imgs.emplace_back(std::move(iter->second));//��move���彫��ŵ��µ�����
			tmpRects.emplace_back(std::move(iter->first));
		}

		model2Handle->processDataConcurrency(imgs);
		std::vector<model2Result> tempResults = model2Handle->m_results;
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
	//���ǽ�model2PS�ŵ�rResults��
	for (auto iter = model2PS.begin(); iter != model2PS.end(); iter++) {
		cv::Point childPoint = iter->point;//�����model2�����Ͻǵ�
		for (int i = 0; i < placeStop; i++) {
			for (int j = 1; j < rResults[i].result.points.size(); j++) {
				cv::Point fatherPoint = rResults[i].result.points[j];
				fatherPoint.x = rResults[i].point.x + fatherPoint.x * float(model1Mpp / slideMpp) - model2Height * float(model2Mpp / slideMpp) / 2;//�������model2��ȫ�ֶ�λ��
				fatherPoint.y = rResults[i].point.y + fatherPoint.y * float(model1Mpp / slideMpp) - model2Width * float(model2Mpp / slideMpp) / 2;
				if (fatherPoint == childPoint) {
					//�ҵ�����������ͬ����score2��־λ��Ϊtrue
					if (!inFlag[i][j - 1]) {
						//��model2PS�ķ�����������
						rResults[i].score2[j - 1] = iter->score;
					}
				}
			}
		}
	}
}