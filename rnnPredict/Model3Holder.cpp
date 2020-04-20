#include "Model3Holder.h"

Model3Holder::Model3Holder()
{}

Model3Holder::Model3Holder(string model3Path)
{
	model3Config(model3Path);
}

Model3Holder::~Model3Holder()
{
	delete model3Handle;
}

void Model3Holder::initPara(MultiImageRead& mImgRead)
{
	mImgRead.getSlideHeight(slideHeight);
	mImgRead.getSlideWidth(slideWidth);
	mImgRead.getSlideMpp(slideMpp);
}

void Model3Holder::model3Config(string model3Path)
{
	modelConfig conf;
	conf.height = 256;
	conf.width = 256;
	conf.channel = 3;
	conf.opsInput = "input_1:0";
	conf.opsOutput.emplace_back("last_dense_output/Softmax:0");
	std::ifstream file(model3Path, std::ios::binary | std::ios::ate);
	std::streamsize size = file.tellg();
	std::unique_ptr<char[]> uBuffer(new char[size]);
	file.seekg(0, std::ios::beg);
	if (!file.read(uBuffer.get(), size)) {
		std::cout << "read file to buffer failed" << endl;
	}
	model3Handle = new model3(conf, uBuffer.get(), size);

	model3Height = 256;
	model3Width = 256;
	model3Mpp = 0.293f;
}

bool Model3Holder::popModel2Queue(vector<std::pair<cv::Rect, cv::Mat>>& rectMats)
{
	//先加锁
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
		//如果这个enterQueue1线程没有退出，那么就开始进行wait操作
		queue_cv2.wait_for(m2Guard, 3000ms, [this] {
			if (model2Queue.size() > 0)
				return true;
			return false;
			});
		//取到锁之后，再次进行循环
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

void Model3Holder::enterModel2Queue(MultiImageRead& mImgRead)
{
	enterFlag2 = true;
	vector<std::pair<cv::Rect, cv::Mat>> tempRectMats;
	while (mImgRead.popQueue(tempRectMats)) {
		for (auto iter = tempRectMats.begin(); iter != tempRectMats.end(); iter++) {
			if (iter->second.cols != model3Width) {
				cv::resize(iter->second, iter->second, cv::Size(model3Width, model3Height));
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

cv::Point Model3Holder::rect2Point(int x, int y, float radius)
{
	cv::Point point(ceil(x + radius), ceil(y + radius));
	return point;
}

vector<PointScore> Model3Holder::model3Recom(vector<std::pair<cv::Rect, model3Result>>& xyResults)
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
	vector<PointScore> retPs;
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

vector<PointScore> Model3Holder::runModel3(MultiImageRead& mImgRead, vector<Anno> &annos)
{
	initPara(mImgRead);
	mImgRead.setGammaFlag(false);
	vector<cv::Rect> rects;
	for (int i = 0; i < annos.size(); i++) {
		cv::Rect rect;
		rect.x = annos[i].x - model3Width / 2 * float(model3Mpp / slideMpp);
		rect.y = annos[i].y - model3Height / 2 * float(model3Mpp / slideMpp);
		rect.height = model3Height * float(model3Mpp / slideMpp);
		rect.width = model3Width * float(model3Mpp / slideMpp);
		rects.emplace_back(rect);
	}
	mImgRead.setRects(rects);
	std::thread threadEnterQueue(&Model3Holder::enterModel2Queue, this, std::ref(mImgRead));
	while (enterFlag2 != true) {
		continue;
	}
	vector<std::pair<cv::Rect, cv::Mat>> rectMats;
	vector<std::pair<cv::Rect, cv::Mat>> tmpRectMats;
	while (popModel2Queue(tmpRectMats)) {
		for (auto iter = tmpRectMats.begin(); iter != tmpRectMats.end(); iter++) {
			rectMats.emplace_back(std::move(*iter));
		}
		tmpRectMats.clear();
	}
	threadEnterQueue.join();
	vector<cv::Mat> imgs;
	vector<std::pair<cv::Rect, model3Result>> xyResults;
	for (auto& elem : rectMats)
	{
		imgs.emplace_back(std::move(elem.second));
	}
	//开始预测50张图像
	vector<model3Result> results = model3Handle->model3Process(imgs);
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