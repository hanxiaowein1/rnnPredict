#include "Model2Holder.h"

Model2Holder::Model2Holder()
{
}

Model2Holder::Model2Holder(string model2Path)
{
	model2Config(model2Path);
}

Model2Holder::~Model2Holder()
{
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

void Model2Holder::model2Config(string model2Path)
{
	modelConfig conf;
	conf.height = 256;
	conf.width = 256;
	conf.channel = 3;
	conf.opsInput = "input_1:0";
	conf.opsOutput.emplace_back("dense_2/Sigmoid:0");
	conf.opsOutput.emplace_back("global_max_pooling2d_1/Max:0");

	std::ifstream file(model2Path, std::ios::binary | std::ios::ate);
	std::streamsize size = file.tellg();
	std::unique_ptr<char[]> uBuffer(new char[size]);
	file.seekg(0, std::ios::beg);
	if (!file.read(uBuffer.get(), size)) {
		std::cout << "read file to buffer failed" << endl;
	}
	model2Handle = new model2(conf, uBuffer.get(), size);
	model2Mpp = model2Handle->getM2Res();
	model2Height = conf.height;
	model2Width = conf.width;
}

void Model2Holder::enterModel2Queue(MultiImageRead& mImgRead)
{
	enterFlag2 = true;
	vector<std::pair<cv::Rect, cv::Mat>> tempRectMats;
	while (mImgRead.popQueue(tempRectMats)) {
		for (auto iter = tempRectMats.begin(); iter != tempRectMats.end(); iter++) {
			if (iter->second.cols != model2Width) {
				cv::resize(iter->second, iter->second, cv::Size(model2Width, model2Height));
			}
		}
		std::unique_lock<std::mutex> m2Guard(queue2Lock);
		for (auto iter = tempRectMats.begin(); iter != tempRectMats.end(); iter++) {
			model2Queue.emplace(std::move(*iter));
		}
		m2Guard.unlock();
		queue_cv2.notify_one();
		tempRectMats.clear();
	}
	enterFlag2 = false;
}

bool Model2Holder::popModel2Queue(vector<std::pair<cv::Rect, cv::Mat>>& rectMats)
{
	//�ȼ���
	std::unique_lock < std::mutex > m2Guard(queue2Lock);
	if (model2Queue.size() > 0) {
		int size = model2Queue.size();
		for (int i = 0; i < size; i++) {
			rectMats.emplace_back(std::move(model2Queue.front()));
			model2Queue.pop();
		}
		return true;
	}
	else if (enterFlag2.load()) {
		//������enterQueue1�߳�û���˳�����ô�Ϳ�ʼ����wait����
		queue_cv2.wait_for(m2Guard, 3000ms, [this] {
			if (model2Queue.size() > 0)
				return true;
			return false;
			});
		//ȡ����֮���ٴν���ѭ��
		int size = model2Queue.size();
		if (size == 0 && enterFlag2.load()) {
			m2Guard.unlock();
			bool flag = popModel2Queue(rectMats);
			return flag;
		}
		if (size == 0 && !enterFlag2.load()) {
			return false;
		}
		for (int i = 0; i < size; i++) {
			rectMats.emplace_back(std::move(model2Queue.front()));
			model2Queue.pop();
		}
		return true;
	}
	else {
		return false;
	}
}

void Model2Holder::sortResultsByScore(vector<regionResult>& results)
{
	auto lambda = [](regionResult result1, regionResult result2)->bool {
		if (result1.result.score > result2.result.score)
			return true;
		return false;
	};
	std::sort(results.begin(), results.end(), lambda);
}

void Model2Holder::model2Process(vector<cv::Mat>& imgs, vector<Tensor>& tensors)
{
	model2Handle->model2Process(imgs, tensors);
}

void Model2Holder::runModel2(MultiImageRead& mImgRead, vector<regionResult>& rResults)
{
	initPara(mImgRead);
	//��rResults������ѡ����model2�Ŀ�
	//���շ����Ӵ�С����
	int m2MinNum = 600;
	int m2MaxNum = 1200;

	sortResultsByScore(rResults);
	vector<cv::Rect> rects;
	int placeStop = 0;

	//�ҿ���һ��ʼ�ͽ�model1��λ��Խ���ȫ���ɵ���...
	//��copyһ�����������rResults��Ȼ���ٴӸ�������������ķŵ�rResults����
	vector<regionResult> rResultsCP = rResults;
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
	mImgRead.setQueueMaxNum(rects.size());
	mImgRead.setReadLevel(0);//���㲻�䣬model2һֱ���Ǵ�level0�Ͻ��ж�ȡ
	mImgRead.setRects(rects);

	//std::thread thread1(&SlideProc::enterModel2Queue2, this, std::ref(enterFlag7), std::ref(mImgRead));
	//std::thread thread2(&SlideProc::enterModel2Queue2, this, std::ref(enterFlag8), std::ref(mImgRead));
	//std::thread thread3(&SlideProc::enterModel2Queue2, this, std::ref(enterFlag9), std::ref(mImgRead));
	//std::thread thread4(&SlideProc::enterModel2Queue2, this, std::ref(enterFlag10), std::ref(mImgRead));
	//while (!checkFlags2())
	//{
	//	continue;
	//}
	//vector<PointScore> model2PS;
	//std::vector<std::pair<vector<cv::Rect>, Tensor>> rectsTensors;
	//while (popModel2Queue(rectsTensors))
	//{
	//	vector<Tensor> tensors;
	//	vector<cv::Rect> tmpRects;
	//	int size = rectsTensors.size();
	//	for (auto iter = rectsTensors.begin(); iter != rectsTensors.end(); iter++)
	//	{
	//		tensors.emplace_back(std::move(iter->second));
	//		tmpRects.insert(tmpRects.end(), iter->first.begin(), iter->first.end());
	//	}
	//	vector<float> scores = model2Handle->model2Process(tensors);
	//	for (int i = 0; i < tmpRects.size(); i++)
	//	{
	//		PointScore ps;
	//		cv::Point point;
	//		point.x = tmpRects[i].x;
	//		point.y = tmpRects[i].y;
	//		ps.point = point;
	//		ps.score = scores[i];
	//		model2PS.emplace_back(ps);
	//	}
	//	rectsTensors.clear();
	//}
	//thread1.join();
	//thread2.join();
	//thread3.join();
	//thread4.join();

	std::thread threadEnterQueue(&Model2Holder::enterModel2Queue, this, std::ref(mImgRead));
	while (enterFlag2 != true)
	{
		continue;
	}
	//Sleep(3000);
	std::vector<std::pair<cv::Rect, cv::Mat>> rectMats;
	vector<PointScore> model2PS;
	//int count = 0;
	while (popModel2Queue(rectMats)/*mImgRead.popQueue(rectMats)*/)
	{
		vector<cv::Mat> imgs;
		vector<cv::Rect> tmpRects;
		for (auto iter = rectMats.begin(); iter != rectMats.end(); iter++) {
			imgs.emplace_back(std::move(iter->second));//��move���彫��ŵ��µ�����
			tmpRects.emplace_back(std::move(iter->first));
		}
		vector<float> scores = model2Handle->model2Process(imgs);
		for (int i = 0; i < tmpRects.size(); i++) {
			PointScore ps;
			cv::Point point;
			point.x = tmpRects[i].x;
			point.y = tmpRects[i].y;
			ps.point = point;
			ps.score = scores[i];
			model2PS.emplace_back(ps);
		}
		rectMats.clear();
	}
	threadEnterQueue.join();
	vector<vector<bool>> inFlag(placeStop);
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