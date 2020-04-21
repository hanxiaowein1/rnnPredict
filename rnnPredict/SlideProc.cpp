#include "SlideProc.h"

#include <fstream>
#include <numeric>

SlideProc::SlideProc(const char* iniPath)
{
	initialize_handler(iniPath);
	//loadXgdll();
}

SlideProc::~SlideProc()
{
	freeMemory();
}

void SlideProc::freeMemory()
{
	delete m1Holder;
	delete m2Holder;
	delete m3Holder;
	for (int i = 0; i < rnnHandle.size(); i++) {
		delete rnnHandle[i];
	}
}

bool SlideProc::iniPara2(const char* slide, MultiImageRead& mImgRead)
{
	//初始化切片的宽高、mpp、ratio
	mImgRead.getSlideHeight(slideHeight);
	mImgRead.getSlideWidth(slideWidth);
	mImgRead.getSlideMpp(slideMpp);
	string suffix = getFileNameSuffix(string(slide));
	if (suffix == "svs")
		slideMpp = 0.293f;
	if (slideHeight <= 0 || slideWidth <= 0 || slideMpp <= 0)
		return false;
	//再来判断一些不合理的范围
	if (slideHeight > 1000000 || slideWidth > 1000000 || slideMpp > 1000000)
		return false;
	slideRatio = mImgRead.get_ratio();

	//初始化读取model1的level
	read_level = (model1Mpp / slideMpp) / slideRatio;

	//初始化从哪一个level读取binImg
	double mySetMpp = 3.77f;//最原始的读取level4的mpp
	double compLevel = mySetMpp / slideMpp;
	vector<double> mppList;
	while (compLevel > 0.1f) {
		mppList.emplace_back(compLevel);
		compLevel = compLevel / slideRatio;
	}
	//遍历mppList，寻找与1最近的值
	double closestValue = 1000.0f;
	for (int i = 0; i < mppList.size(); i++) {
		if (std::abs(mppList[i] - 1.0f) < closestValue) {
			closestValue = std::abs(mppList[i] - 1.0f);
			levelBin = i;
		}
	}

	//初始化前景分割的阈值
	double mySetMpp2 = 0.235747f;//最原始的前景分割的mpp
	int thre_vol = 150;//最原始的面积阈值，在mySetMpp2上
	m_thre_vol = thre_vol / (slideMpp / mySetMpp2);
	int crop_sum = 960;//最原始的从binImg抠图的求和阈值，在mySetMpp2下
	m_crop_sum = crop_sum / (slideMpp / mySetMpp2);
	m_crop_sum = m_crop_sum / std::pow(slideRatio, levelBin);
	return true;
}

bool SlideProc::iniPara(const char* slide, MultiImageRead& mImgRead)
{
	if (m_srpRead != nullptr) {
		delete m_srpRead;
		m_srpRead = new SrpSlideRead(slide);
	
	}
	else {
		m_srpRead = new SrpSlideRead(slide);
	}
	if (!m_srpRead->status())
		return false;
	//初始化切片的宽高、mpp、ratio
	mImgRead.getSlideHeight(slideHeight);
	mImgRead.getSlideWidth(slideWidth);
	mImgRead.getSlideMpp(slideMpp);
	if (slideHeight <= 0 || slideWidth <= 0 || slideMpp <= 0)
		return false;
	//再来判断一些不合理的范围
	if (slideHeight > 1000000 || slideWidth > 1000000 || slideMpp > 1000000)
		return false;
	slideRatio = mImgRead.get_ratio();

	//初始化读取model1的level
	read_level = (model1Mpp / slideMpp) / slideRatio;

	//初始化从哪一个level读取binImg
	double mySetMpp = 3.77f;//最原始的读取level4的mpp
	double compLevel = mySetMpp / slideMpp;
	vector<double> mppList;
	while (compLevel > 0.1f) {
		mppList.emplace_back(compLevel);
		compLevel = compLevel / slideRatio;
	}
	//遍历mppList，寻找与1最近的值
	double closestValue = 1000.0f;
	for (int i = 0; i < mppList.size(); i++) {
		if (std::abs(mppList[i] - 1.0f) < closestValue) {
			closestValue = std::abs(mppList[i] - 1.0f);
			levelBin = i;
		}
	}

	//初始化前景分割的阈值
	double mySetMpp2 = 0.235747f;//最原始的前景分割的mpp
	int thre_vol = 150;//最原始的面积阈值，在mySetMpp2上
	m_thre_vol = thre_vol / (slideMpp / mySetMpp2);
	int crop_sum = 960;//最原始的从binImg抠图的求和阈值，在mySetMpp2下
	m_crop_sum = crop_sum / (slideMpp / mySetMpp2);
	m_crop_sum = m_crop_sum / std::pow(slideRatio, levelBin);
	return true;
}

vector<cv::Rect> SlideProc::get_rects_slide(MultiImageRead& mImgRead)
{
	vector<cv::Rect> rects;
	int constant1 = std::pow(slideRatio, read_level);

	int read_level_height = slideHeight / constant1;
	int read_level_width = slideWidth / constant1;
	//mImgRead.getLevelDimensions(read_level, read_level_width, read_level_height);
	int crop_width = 8192 / constant1;
	int crop_height = 8192 / constant1;
	if (crop_width > read_level_height || crop_height > read_level_width)
	{
		return rects;
	}

	int sHeight = model1Height * float(model1Mpp / slideMpp);
	int sWidth = model1Width * float(model1Mpp / slideMpp);
	sHeight = sHeight / constant1;
	sWidth = sWidth / constant1;
	//这个overlap要自适应
	//int overlap = 560 / constant1;
	//计算新的overlap
	int overlap_s = sWidth * model1OverlapRatio;
	int n = (crop_width - overlap_s) / (sWidth - overlap_s);
	int overlap = crop_width - n * (sWidth - overlap_s);

	int x_num = (read_level_width - overlap) / (crop_width - overlap);
	int y_num = (read_level_height - overlap) / (crop_height - overlap);
	

	vector<int> xStart;
	vector<int> yStart;
	bool flag_right = true;
	bool flag_down = true;
	if ((x_num * (crop_width - overlap) + overlap) == read_level_width) {
		flag_right = false;
	}
	if ((y_num * (crop_height - overlap) + overlap) == read_level_height) {
		flag_down = false;
	}
	for (int i = 0; i < x_num; i++) {
		xStart.emplace_back((crop_width - overlap) * i);
	}
	for (int i = 0; i < y_num; i++) {
		yStart.emplace_back((crop_height - overlap) * i);
	}
	int last_width = read_level_width - x_num * (crop_width - overlap);
	int last_height = read_level_height - y_num * (crop_height - overlap);
	if (flag_right) {
		if (last_width >= sWidth)
			xStart.emplace_back((crop_width - overlap) * x_num);
		else {
			xStart.emplace_back(read_level_width - sWidth);
			last_width = sWidth;
		}
	}
	if (flag_down) {
		if (last_height >= sHeight)
			yStart.emplace_back((crop_height - overlap) * y_num);
		else {
			yStart.emplace_back(read_level_height - sHeight);
			last_height = sHeight;
		}
	}
	for (int i = 0; i < yStart.size(); i++) {
		for (int j = 0; j < xStart.size(); j++) {
			cv::Rect rect;
			rect.x = xStart[j];
			rect.y = yStart[i];
			rect.width = crop_width;
			rect.height = crop_height;
			if (i == yStart.size() - 1) {
				rect.height = last_height;
			}
			if (j == xStart.size() - 1) {
				rect.width = last_width;
			}
			rects.emplace_back(rect);
		}
	}
	return rects;
}

cv::Point SlideProc::rect2Point(int x, int y, float radius)
{
	cv::Point point(ceil(x + radius), ceil(y + radius));
	return point;
}

cv::Rect SlideProc::point2Rect(int x, int y, float radius, float diameter)
{
	cv::Rect rect(x - radius, y - radius, diameter, diameter);
	return rect;
}

vector<Anno> SlideProc::regionProposal(int recom)
{
	//采取新的推荐方式，按照model2排序，然后去重阈值选择200us，选择10个，不够就从model1从大到小排序进行补充
	int threshold = 200 / slideMpp;
	int tmpRecomSize = recom;
	vector<PointScore> model2PS;
	for (auto &elem:rResults) {
		if (elem.score2.size() > 0) {
			for (auto iter = elem.score2.begin(); iter != elem.score2.end(); iter++) {
				int place = iter - elem.score2.begin();
				PointScore ps;
				ps.score = *iter;
				//ps.point = elem.point .result.points[place + 1]
				ps.point.x = elem.point.x + elem.result.points[place + 1].x * float(model1Mpp / slideMpp);
				ps.point.y = elem.point.y + elem.result.points[place + 1].y * float(model1Mpp / slideMpp);
				model2PS.emplace_back(ps);
			}
		}
	}
	//然后按照分数从大到小进行排序
	sortResultsByScore(model2PS);
	vector<Anno> annos;
	for (auto& elem : model2PS) {
		Anno anno;
		bool flag = false;
		anno.x = elem.point.x;
		anno.y = elem.point.y;
		anno.score = elem.score;
		for (auto& elem2 : annos) {
			//if (std::abs(elem2.x - anno.x) < threshold && std::abs(elem2.y - anno.y) < threshold) {
			//	flag = true;
			//	break;
			//}
			//这个条件改成两点间的距离
			int x_dis = std::abs(elem2.x - anno.x);
			int y_dis = std::abs(elem2.y - anno.y);
			int dis_square = std::pow(x_dis, 2) + std::pow(y_dis, 2);
			if (std::pow(dis_square, 0.5f) < threshold)
			{
				flag = true;
				break;
			}
		}
		if (flag)
			continue;
		annos.emplace_back(anno);
		if (annos.size() >= tmpRecomSize) {
			break;
		}
	}
	//如果推荐的数量不足
	if (annos.size() < tmpRecomSize) {
		vector<regionResult> results = rResults;
		model2PS.clear();
		sortResultsByScore(results);
		for (auto& elem : results) {
			//然后从results里面挑没有model2分数的定位点，并将其分数设置为model1的分数
			if (elem.score2.size() > 0) {
				continue;
			}
			//选取model1的分数
			if (elem.result.points.size() > 1) {
				Anno anno;
				anno.score = elem.result.score;
				anno.x = elem.point.x + elem.result.points[1].x * float(model1Mpp / slideMpp);
				anno.y = elem.point.y + elem.result.points[1].y * float(model1Mpp / slideMpp);
				bool flag = false;
				for (auto& elem2 : annos) {
					if (std::abs(elem2.x - anno.x) < threshold && std::abs(elem2.y - anno.y) < threshold) {
						flag = true;
						break;
					}
				}
				if (flag)
					continue;
				annos.emplace_back(anno);
				if (annos.size() >= tmpRecomSize) {
					break;
				}
			}
		}
	}
	return annos;
}

float SlideProc::runRnnThread2(int i, Tensor& inputTensor)
{
	//对inputTensor要转为1*?*?
	if (inputTensor.dims() != 2)
	{
		cout << "runRnnThread2: inputTensor dims should be 2\n";
		return -1;
	}	
	tensorflow::Tensor tem_tensor_res(
		tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({ 1, inputTensor.dim_size(0), inputTensor.dim_size(1) }));
	std::memcpy(tem_tensor_res.flat<float>().data(), inputTensor.flat<float>().data(), 
		inputTensor.dim_size(0) * inputTensor.dim_size(1) * sizeof(float));
	vector<Tensor> outputTensor;
	rnnHandle[i]->output(tem_tensor_res, outputTensor);
	vector<float> score = rnnHandle[i]->rnnProcess(outputTensor[0]);
	return score[0];
	//return -1;
}

float SlideProc::runRnnThread(int i, Tensor& inputTensor)
{
	//根据i来调用哪一个rnnHandle
	auto model2OutData = inputTensor.tensor<float, 2>();
	Tensor rnnInput(tensorflow::DataType::DT_FLOAT,
		tensorflow::TensorShape({ 1, 10, 2048 }));
	auto rnnInputData = rnnInput.tensor<float, 3>();
	vector<int> order;
	for (int orderCount = 0; orderCount < 10; orderCount++) {
		order.emplace_back(orderCount);
	}
	vector<float> scoreCount;
	for (int loopCount = 0; loopCount < 500; loopCount++) {
		srand(time(0));
		random_shuffle(order.begin(), order.end());
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 2048; j++) {
				rnnInputData(0, order[i], j) = model2OutData(i, j);
			}
		}
		vector<Tensor> rnnOutTensor;
		rnnHandle[i]->output(rnnInput, rnnOutTensor);
		//将rnnOutTensor转为float类型的score
		rnn rnnObj;
		vector<float> score;
		score = rnnHandle[i]->rnnProcess(rnnOutTensor[0]);
		scoreCount.emplace_back(score[0]);
	}
	float sum = std::accumulate(std::begin(scoreCount), std::end(scoreCount), 0.0f);
	float mean = sum / scoreCount.size();
	return mean;
}

float SlideProc::runRnn(vector<Anno>& annos, MultiImageRead& mImgRead)
{
	//传入10个点，然后在全图上进行抠图
	vector < cv::Rect > rects;
	for (int i = 0; i < annos.size(); i++) {
		cv::Rect rect;
		rect.x = annos[i].x - model2Width / 2 * float(model2Mpp / slideMpp);
		rect.y = annos[i].y - model2Height / 2 * float(model2Mpp / slideMpp);
		rect.height = model2Height * float(model2Mpp / slideMpp);
		rect.width = model2Width * float(model2Mpp / slideMpp);
		rects.emplace_back(rect);
	}
	mImgRead.setRects(rects);
	std::thread threadEnterQueue(&SlideProc::enterModel2Queue, this, std::ref(mImgRead));
	while (enterFlag2 != true) {
		continue;
	}
	//Sleep(3000);
	vector<std::pair<cv::Rect, cv::Mat>> rectMats;
	vector<std::pair<cv::Rect, cv::Mat>> tmpRectMats;
	while (popModel2Queue(rectMats)/*mImgRead.popQueue(tmpRectMats)*/) {
		for (auto iter = tmpRectMats.begin(); iter != tmpRectMats.end(); iter++) {
			rectMats.emplace_back(std::move(*iter));
		}
		tmpRectMats.clear();
	}
	threadEnterQueue.join();
	//一个重点的错误！！rnn进入的tensor应该是从大到小的
	//无需flags，因为去重之后坐标不可能相同
	vector<bool> flags(annos.size(), false);
	vector<std::pair<cv::Rect, cv::Mat>> rectMats2(rectMats.size());
	
	for (int i = 0; i < rectMats.size(); i++)
	{
		for (int j = 0; j < annos.size(); j++)
		{
			if (rectMats[i].first.x == rects[j].x && rectMats[i].first.y == rects[j].y)
			{
				rectMats2[j] = std::move(rectMats[i]);
				break;
			}
		}
	}


	//将10张图像放到model2中进行预测得到tensor
	vector<Tensor> tensors;
	vector<cv::Mat> imgs;
	for (int i = 0; i < rectMats.size(); i++) {
		imgs.emplace_back(std::move(rectMats2[i].second));
	}
	m2Holder->model2Process(imgs, tensors);

	float retScore = rnnHolder->runRnn(tensors[1]);
	return retScore;

	//需要修改为3个rnn模型
	//tensor[1]，变成了30*2048个向量，需要对其splice
	//vector<Tensor> rnnInputTensor;
	//Tensor tensor10 = tensors[1].Slice(0, 10);
	//Tensor tensor20 = tensors[1].Slice(0, 20);
	//rnnInputTensor.emplace_back(tensor10);
	//rnnInputTensor.emplace_back(tensor20);
	//rnnInputTensor.emplace_back(tensors[1]);

	//vector<std::future<float>> rnnResults(rnnHandle.size());
	//for (int i = 0; i < rnnHandle.size(); i++)
	//{
	//	rnnResults[i] = std::async(&SlideProc::runRnnThread2, this, i, std::ref(rnnInputTensor[i / 2]));
	//}
	//vector<float> rnnResults_f;
	//for (int i = 0; i < rnnResults.size(); i++)
	//{
	//	rnnResults_f.emplace_back(rnnResults[i].get());
	//}
	//cout << "rnnResults_f: ";
	//for (int i = 0; i < rnnResults_f.size(); i++)
	//{
	//	cout << rnnResults_f[i] << " ";
	//}
	//cout << endl;
	//if (rnnResults_f.size() != 6)
	//	return -1;
	////先求6个数的平均值
	//float sum_6 = std::accumulate(rnnResults_f.begin(), rnnResults_f.end(), 0.0f);
	//float avg_6 = sum_6 / rnnResults_f.size();
	//float avg2_1 = std::accumulate(rnnResults_f.begin(), rnnResults_f.begin() + 2, 0.0f) / 2;//前两个平均值
	//float avg2_2 = std::accumulate(rnnResults_f.begin() + 2, rnnResults_f.begin() + 4, 0.0f) / 2;//...
	//float avg2_3 = std::accumulate(rnnResults_f.begin() + 4, rnnResults_f.end(), 0.0f) / 2;//...
	//vector<float> avg3_total{ avg2_1, avg2_2, avg2_3 };
	//float max_3 = *std::max_element(avg3_total.begin(), avg3_total.end());
	//float min_3 = *std::min_element(avg3_total.begin(), avg3_total.end());

	//float sum_3 = std::accumulate(avg3_total.begin(), avg3_total.end(), 0.0f);
	//float avg_3 = sum_3 / avg3_total.size();
	//float accum = 0.0;
	//std::for_each(avg3_total.begin(), avg3_total.end(), [&](const float d) {
	//	accum += (d - avg_3) * (d - avg_3);
	//	});
	//accum = accum / avg3_total.size();//方差
	//float std_3 = std::pow(accum, 0.5f);

	//float retScore = 0.0f;
	//if (std_3 < 0.15f)
	//{
	//	if (avg_6 < 0.15f)
	//		retScore = min_3;
	//	else
	//		retScore = max_3;
	//}
	//else
	//	retScore = avg_6;
	//return retScore;
}

void SlideProc::sortResultsByCoor(vector<regionResult>& results)
{
	auto lambda = [](regionResult result1, regionResult result2)->bool {
		if (result1.point.y < result2.point.y) {
			return true;
		}
		else if (result1.point.y > result2.point.y) {
			return false;
		}
		else {
			if (result1.point.x < result2.point.x) {
				return true;
			}
			return false;
		}
	};
	std::sort(results.begin(), results.end(), lambda);
}

void SlideProc::removeDuplicatePS(vector<PointScore>& pss1, vector<PointScore>& pss2, int threshold)
{
	//1.对pss1去重
	filterBaseOnPoint(pss1, threshold);
	//2.对pss2去重
	filterBaseOnPoint(pss2, threshold);
	//3.对pss1和pss2去重
	pss1.insert(pss1.end(), pss2.begin(), pss2.end());
	filterBaseOnPoint(pss1, threshold);
}

void SlideProc::filterBaseOnPoint(vector<PointScore>& PointScores, int threshold)
{
	vector<PointScore> psCopy;
	psCopy.insert(psCopy.end(), PointScores.begin(), PointScores.end());
	PointScores.clear();
	for (auto iter = psCopy.begin(); iter != psCopy.end(); iter++) {
		Point point = iter->point;
		int place = iter - psCopy.begin();
		bool flag = true;
		for (auto iter2 = PointScores.begin(); iter2 != PointScores.end(); iter2++) {
			Point point2 = iter2->point;
			int x_dis = std::abs(point.x - point2.x);
			int y_dis = std::abs(point.y - point2.y);
			int dis_square = std::pow(x_dis, 2) + std::pow(y_dis, 2);
			if (std::pow(dis_square, 0.5f) < threshold)
			{
				flag = false;
				break;
			}
			//if (abs(point.x - point2.x) < threshold && abs(point.y - point2.y) < threshold)
			//{
			//	flag = false;
			//	break;
			//}
		}
		if (flag) {
			PointScores.emplace_back(*iter);
		}
	}
}

void SlideProc::sortResultsByScore(vector<PointScore>& pss)
{
	auto lambda = [](PointScore ps1, PointScore ps2)->bool {
		if (ps1.score > ps2.score) {
			return true;
		}
		return false;
	};
	std::sort(pss.begin(), pss.end(), lambda);
}

void SlideProc::sortResultsByScore(vector<regionResult> &results)
{
	auto lambda = [](regionResult result1, regionResult result2)->bool {
		if (result1.result.score > result2.result.score)
			return true;
		return false;
	};
	std::sort(results.begin(), results.end(), lambda);
}

//
//void SlideProc::enterModel2Queue2(std::atomic<bool>& flag, MultiImageRead& mImgRead)
//{
//	//接下来的步骤和model1的步骤相同
//	flag = true;
//	vector<std::pair<cv::Rect, cv::Mat>> rectMats;
//	while (mImgRead.popQueue(rectMats)) {
//		for (auto iter = rectMats.begin(); iter != rectMats.end(); iter++) {
//			cv::resize(iter->second, iter->second, Size(model2Height, model2Width));
//		}
//		vector<std::pair<vector<cv::Rect>, Tensor>> rectsTensors;
//		Mats2Tensors(rectMats, rectsTensors, model2_batchsize);
//		std::unique_lock<std::mutex> m2Guard(tensor_queue_lock2);
//		for (auto iter = rectsTensors.begin(); iter != rectsTensors.end(); iter++) {
//			tensor_queue2.emplace(std::move(*iter));
//		}
//		m2Guard.unlock();
//		tensor_queue_cv2.notify_one();
//		rectMats.clear();
//	}
//	flag = false;
//}

void SlideProc::enterModel2Queue(MultiImageRead& mImgRead)
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

bool SlideProc::checkFlags2()
{
	if (enterFlag7.load() || enterFlag8.load() || enterFlag9.load() || enterFlag10.load())
		return true;
	return false;
}

bool SlideProc::popModel2Queue(vector<std::pair<vector<cv::Rect>, Tensor>>& rectsTensors)
{
	std::unique_lock < std::mutex > m2Guard(tensor_queue_lock2);
	if (tensor_queue2.size() > 0) {
		int size = tensor_queue2.size();
		for (int i = 0; i < size; i++) {
			rectsTensors.emplace_back(std::move(tensor_queue2.front()));
			tensor_queue2.pop();
		}
		return true;
	}
	else if (checkFlags2()) {
		tensor_queue_cv2.wait_for(m2Guard, 3000ms, [this] {
			if (tensor_queue2.size() > 0)
				return true;
			return false;
			});
		int size = tensor_queue2.size();
		if (size == 0 && checkFlags2()) {
			m2Guard.unlock();
			bool flag = popModel2Queue(rectsTensors);
			return flag;
		}
		if (size == 0 && !checkFlags2()) {
			return false;
		}
		for (int i = 0; i < size; i++) {
			rectsTensors.emplace_back(std::move(tensor_queue2.front()));
			tensor_queue2.pop();
		}
		return true;
	}
	else {
		return false;
	}
}

bool SlideProc::popModel2Queue(vector<std::pair<cv::Rect, cv::Mat>>& rectMats)
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

void SlideProc::saveResult3(string savePath, string saveName)
{
	std::ofstream out(savePath + saveName + ".txt");
	string saveString = to_string(slideScore);
	out << saveString;
	out.close();
}

void SlideProc::saveResult2(string savePath, string saveName)
{
	vector<PointScore> model2PS;
	for (auto& elem : rResults) {
		if (elem.score2.size() > 0) {
			for (auto iter = elem.score2.begin(); iter != elem.score2.end(); iter++) {
				int place = iter - elem.score2.begin();
				PointScore ps;
				ps.score = *iter;
				//ps.point = elem.point .result.points[place + 1]
				ps.point.x = elem.point.x + elem.result.points[place + 1].x * float(model1Mpp / slideMpp);
				ps.point.y = elem.point.y + elem.result.points[place + 1].y * float(model1Mpp / slideMpp);
				model2PS.emplace_back(ps);
			}
		}
	}
	int threshold = 200 / slideMpp;	
	sortResultsByScore(model2PS);
	filterBaseOnPoint(model2PS, threshold);
	//sortResultsByScore(model2PS);
	std::ofstream out(savePath + "\\"+saveName + ".txt");
	string saveString = "";
	for (auto& elem : model2PS) {
		saveString = saveString + to_string(elem.score) + "," + to_string(elem.point.x) + "," + to_string(elem.point.y) + "\n";
	}
	out << saveString;
	out.close();
}

void SlideProc::saveImages(vector<PointScore>& pss, int radius, string savePath)
{
	//直接用m_srpRead来进行有序的读取(因为不是大量的数据，不需要开启多线程)
	int i = 0;
	for (auto elem : pss)
	{
		cv::Mat img;
		//在这里判断边界问题，如果小于零，就归为0处理
		cv::Point point;
		if (elem.point.x - radius < 0)
			point.x = 0;
		else
			point.x = elem.point.x - radius;
		if (elem.point.y - radius < 0)
			point.y = 0;
		else
			point.y = elem.point.y - radius;
		m_srpRead->getTile(0, point.x, point.y, radius * 2, radius * 2, img);
		string saveName = to_string(i) + "_" + to_string(elem.point.x) + "_" +
			to_string(elem.point.y) + "_" + to_string(elem.score) + ".tif";
		cv::imwrite(savePath + "\\" + saveName, img);
		i++;
	}
}

vector<PointScore> SlideProc::anno2PS(vector<Anno>& annos)
{
	vector<PointScore> ret;
	for (auto elem : annos)
	{
		PointScore ps;
		ps.point.x = elem.x;
		ps.point.y = elem.y;
		ps.score = elem.score;
		ret.emplace_back(ps);
	}
	return ret;
}

bool SlideProc::runSlide3(const char* slide, string in_savePath)
{
	//初始化片子的信息
	MultiImageRead mImgRead(slide);
	mImgRead.createThreadPool();
	mImgRead.setAddTaskThread();
	if (!mImgRead.status())
		return false;
	iniPara(slide, mImgRead);

	m_slide = string(slide);
	rResults.clear();

	if (slideHeight == 0 || slideWidth == 0 || slideMpp == 0)
		return false;

	time_t now = time(0);
	cout << "start model1 process: " << (char*)ctime(&now);
	//runModel1(mImgRead);
	rResults = m1Holder->runModel1(mImgRead);

	sortResultsByCoor(rResults);
	now = time(0);
	cout << "start model2 process: " << (char*)ctime(&now);
	//runModel2(mImgRead);
	m2Holder->runModel2(mImgRead, rResults);
	sortResultsByCoor(rResults);

	vector<Anno> annos = regionProposal(30);
	slideScore = runRnn(annos, mImgRead);

	//在这里测试一下model3
	vector<Anno> annos_for_m3 = regionProposal(50);
	vector<PointScore> m3Results = m3Holder->runModel3(mImgRead, annos_for_m3);
	//vector<PointScore> m3Results = runModel3(mImgRead);
	if (m3Results.size() > 10)
		m3Results.erase(m3Results.begin() + 10, m3Results.end());
	else
	{
		//先将annos转为PointScore
		vector<PointScore> m2Results = anno2PS(annos);
		int threshold = 200 / slideMpp;
		removeDuplicatePS(m3Results, m2Results, threshold);
		if(m3Results.size()>10)
			m3Results.erase(m3Results.begin() + 10, m3Results.end());
	}
	//将model3和model2的图像全部都保存下来进行查看
	string model2ResultPath = in_savePath + "\\model2";
	string model3ResultPath = in_savePath + "\\model3";
	string model3InputPath = in_savePath + "\\model3Input";
	createDirRecursive(model2ResultPath);
	createDirRecursive(model3ResultPath);
	createDirRecursive(model3InputPath);
	annos.clear();
	annos = regionProposal(10);
	vector<PointScore> pss = anno2PS(annos);
	saveImages(pss, 200, model2ResultPath);
	annos.clear();
	annos = regionProposal(50);
	pss = anno2PS(annos);
	saveImages(pss, 200, model3InputPath);//可能会越界
	saveImages(m3Results, 200, model3ResultPath);//可能会越界
	mImgRead.~MultiImageRead();
}

//
//bool SlideProc::runSlide2(const char* slide)
//{
//	levelBin = 4;
//	MultiImageRead mImgRead(slide);
//	mImgRead.createThreadPool();
//	mImgRead.setAddTaskThread();
//	if (!mImgRead.status())
//		return false;
//	iniPara2(slide, mImgRead);
//	string suffix = getFileNameSuffix(string(slide));
//	if(suffix == "srp") {
//		if (m_srpRead != nullptr) {
//			delete m_srpRead;
//			m_srpRead = new SrpSlideRead(slide);
//		}
//		else {
//			m_srpRead = new SrpSlideRead(slide);
//		}
//		initialize_binImg();
//		mImgRead.setGammaFlag(true);
//	}
//	if (suffix == "sdpc") {
//		if (m_sdpcRead != nullptr) {
//			delete m_sdpcRead;
//			m_sdpcRead = new SdpcSlideRead(slide);
//		}
//		else {
//			m_sdpcRead = new SdpcSlideRead(slide);
//		}
//
//		int heightL4 = 0;
//		int widthL4 = 0;
//		m_sdpcRead->getLevelDimensions(levelBin, widthL4, heightL4);
//		if (widthL4 == 0 || heightL4 == 0) {
//			cout << "get L4 image failed\n";
//			return false;
//		}
//		m_sdpcRead->getTile(levelBin, 0, 0, widthL4, heightL4, imgL4);
//		threshold_segmentation(imgL4, binImg, levelBin, 20, 150);
//		mImgRead.setGammaFlag(true);
//	}
//	if (suffix == "svs" || suffix == "mrxs") {
//		if (m_osRead != nullptr) {
//			delete m_osRead;
//			m_osRead = new OpenSlideRead(slide);
//		}
//		else {
//			m_osRead = new OpenSlideRead(slide);
//		}
//		if (suffix == "svs") {
//
//			slideMpp = 0.293f;
//			m_osRead->getLevelDimensions(0, slideWidth, slideHeight);
//		}
//
//		m_osRead->getTile(
//			levelBin, 0, 0,
//			slideWidth / std::pow(slideRatio, levelBin), slideHeight / std::pow(slideRatio, levelBin), imgL4);
//		threshold_segmentation(imgL4, binImg, levelBin, 20, 150);
//		mImgRead.setGammaFlag(false);		
//	}
//	m_slide = string(slide);
//	rResults.clear();
//	if (slideHeight == 0 || slideWidth == 0 || slideMpp == 0)
//		return false;
//	time_t now = time(0);
//	cout << "start model1 process: " << (char*)ctime(&now);
//	runModel1(mImgRead);
//	now = time(0);
//	cout << "start model2 process: " << (char*)ctime(&now);
//	runModel2(mImgRead);
//	sortResultsByCoor(rResults);
//	return true;
//}
//
//bool SlideProc::runSlide(const char* slide, vector<Anno>& annos)
//{
//	//levelBin = 4;
//
//	//初始化片子的信息
//	MultiImageRead mImgRead(slide);
//	mImgRead.createThreadPool();
//	mImgRead.setAddTaskThread();
//	if (!mImgRead.status())
//		return false;
//	iniPara(slide, mImgRead);
//	bool flag = initialize_binImg();
//	if (!flag)
//		return false;
//
//	m_slide = string(slide);
//	rResults.clear();
//
//	if (slideHeight == 0 || slideWidth == 0 || slideMpp == 0)
//		return false;
//
//	time_t now = time(0);
//	cout << "start model1 process: " << (char*)ctime(&now);
//	runModel1(mImgRead);
//	sortResultsByCoor(rResults);
//	now = time(0);
//	cout << "start model2 process: " << (char*)ctime(&now);
//	runModel2(mImgRead);
//	sortResultsByCoor(rResults);
//	now = time(0);
//	cout << "start regionProposal: " << (char*)ctime(&now);
//	annos = regionProposal(recomNum);
//
//	now = time(0);
//	cout << "start runRnn: " << (char*)ctime(&now);
//	float rnnResult = runRnn(annos, mImgRead);
//	//float xgResult = runXgboost();
//	sortResultsByCoor(rResults);
//
//	slideScore = rnnResult;
//
//	mImgRead.~MultiImageRead();
//
//	//将信息写到srp里面
//	annos.erase(annos.begin() + 10, annos.end());
//	Anno* pann = new Anno[annos.size()];
//	for (int i = 0; i < annos.size(); i++) {
//		pann[i].id = i;
//		pann[i].type = 0;
//		pann[i].x = annos[i].x;
//		pann[i].y = annos[i].y;
//		pann[i].score = annos[i].score;
//	}
//	now = time(0);
//	cout << "write anno to slide: " << (char*)ctime(&now);
//	m_srpRead->callCleanAnno();
//	m_srpRead->callWriteAnno(pann, annos.size());
//	m_srpRead->callWriteParamDouble("score", slideScore);
//
//	mImgRead.~MultiImageRead();
//	if (m_srpRead != nullptr)
//	{
//		delete m_srpRead;
//		m_srpRead = nullptr;
//	}
//
//
//	return true;
//}
