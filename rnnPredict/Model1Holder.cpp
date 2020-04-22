#include "Model1Holder.h"

Model1Holder::Model1Holder(string iniPath)
{
	model1Config(iniPath);
}

Model1Holder::~Model1Holder()
{
	delete model1Handle;
}

bool Model1Holder::iniPara(MultiImageRead& mImgRead)
{
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

void Model1Holder::model1Config(string iniPath)
{
	model1Handle = new TfModel1(iniPath, "TfModel1");
	model1Handle->createThreadPool();
	model1Mpp = model1Handle->inputProp.mpp;
	model1Height = model1Handle->inputProp.height;
	model1Width = model1Handle->inputProp.width;
}

void Model1Holder::normalize(std::vector<cv::Mat>& imgs, tensorflow::Tensor& tensor)
{
	if (imgs.size() == 0)
		return;
	for (int i = 1; i < imgs.size(); i++) {
		if (imgs[i - 1].rows != imgs[i].rows || imgs[i - 1].cols != imgs[i].cols)
			return;
	}
	int tensorHeight = imgs[0].rows;
	int tensorWidth = imgs[0].cols;
	for (int i = 0; i < imgs.size(); i++) {
		float* ptr = tensor.flat<float>().data() + i * tensorHeight * tensorWidth * 3;
		cv::Mat tensor_image(tensorHeight, tensorWidth, CV_32FC3, ptr);
		imgs[i].convertTo(tensor_image, CV_32F);//转为float类型的数组
		tensor_image = (tensor_image / 255 - 0.5) * 2;
	}
}

void Model1Holder::Mats2Tensors(std::vector<std::pair<cv::Rect, cv::Mat>>& rectMats, std::vector<std::pair<std::vector<cv::Rect>, tensorflow::Tensor>>& rectsTensors, int batchsize)
{
	if (rectMats.size() == 0)
		return;
	std::vector<cv::Rect> rects;
	std::vector<cv::Mat> imgs;
	for (auto iter = rectMats.begin(); iter != rectMats.end(); iter++) {
		rects.emplace_back(std::move(iter->first));
		imgs.emplace_back(std::move(iter->second));
	}
	for (int i = 1; i < imgs.size(); i++) {
		if (imgs[i - 1].rows != imgs[i].rows || imgs[i - 1].cols != imgs[i].cols)
			return;
	}
	std::vector<tensorflow::Tensor> tensors;
	Mats2Tensors(imgs, tensors, batchsize);
	//现在将rects和tensors按位置放到rectsTensors里面
	int start = 0;
	int tensorPlace = 0;

	for (int i = 0; i < rects.size(); i = i + batchsize) {
		std::pair<std::vector<cv::Rect>, tensorflow::Tensor> rectsTensor;
		auto iterBegin = rects.begin() + start;
		auto iterEnd = rects.end();
		if (iterBegin + batchsize >= iterEnd) {
			std::vector<cv::Rect> tempRects(iterBegin, iterEnd);
			rectsTensor.first = tempRects;
			rectsTensor.second = std::move(tensors[tensorPlace]);
		}
		else {
			iterEnd = iterBegin + batchsize;
			std::vector<cv::Rect> tempRects(iterBegin, iterEnd);
			rectsTensor.first = tempRects;
			rectsTensor.second = std::move(tensors[tensorPlace]);
			start = start + batchsize;
			tensorPlace++;
		}
		rectsTensors.emplace_back(std::move(rectsTensor));
	}
}

void Model1Holder::Mats2Tensors(std::vector<cv::Mat>& imgs, std::vector<tensorflow::Tensor>& tensors, int batchsize)
{
	int start = 0;
	if (imgs.size() == 0)
		return;
	for (int i = 1; i < imgs.size(); i++) {
		if (imgs[i - 1].rows != imgs[i].rows || imgs[i - 1].cols != imgs[i].cols)
			return;
	}
	int tensorHeight = imgs[0].rows;
	int tensorWidth = imgs[0].cols;
	for (int i = 0; i < imgs.size(); i = i + batchsize) {
		auto iterBegin = imgs.begin() + start;
		std::vector<cv::Mat>::iterator iterEnd = imgs.end();
		if (iterBegin + batchsize >= iterEnd) {
			std::vector<cv::Mat> tempImgs(iterBegin, iterEnd);
			tensorflow::Tensor tensor(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({ iterEnd - iterBegin, tensorHeight, tensorWidth, 3 }));
			normalize(tempImgs, tensor);
			tensors.emplace_back(std::move(tensor));
		}
		else {
			iterEnd = iterBegin + batchsize;
			std::vector<cv::Mat> tempImgs(iterBegin, iterEnd);
			tensorflow::Tensor tensor(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({ iterEnd - iterBegin, tensorHeight, tensorWidth, 3 }));
			normalize(tempImgs, tensor);
			tensors.emplace_back(std::move(tensor));
			start = i + batchsize;
		}
	}
}

void Model1Holder::remove_small_objects(cv::Mat& binImg, int thre_vol)
{
	//去除img中小的区域
	vector<vector<cv::Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(binImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	double threshold = thre_vol;//面积的阈值
	vector<vector<cv::Point>> finalContours;
	for (int i = 0; i < contours.size(); i++) {
		double area = cv::contourArea(contours[i]);
		if (area >= threshold) {
			finalContours.emplace_back(contours[i]);
		}
	}
	if (finalContours.size() > 0) {
		cv::Mat finalMat(binImg.rows, binImg.cols, CV_8UC1, Scalar(0));
		cv::fillPoly(finalMat, finalContours, Scalar(255));
		binImg = finalMat.clone();
	}
}

void Model1Holder::threshold_segmentation(cv::Mat& img, cv::Mat& binImg, int level, int thre_col, int thre_vol)
{
	//对img进行遍历，每三个unsigned char类型，选择其中的最大最小值
	std::unique_ptr<unsigned char[]> pBinBuf(new unsigned char[img.cols * img.rows]);
	unsigned char* pStart = (unsigned char*)img.datastart;
	unsigned char* pEnd = (unsigned char*)img.dataend;
	for (unsigned char* start = pStart; start < pEnd; start = start + 3)
	{
		//选择rgb元素中的最大最小值
		unsigned char R = *start;
		unsigned char G = *(start + 1);
		unsigned char B = *(start + 2);
		unsigned char maxValue = R;
		unsigned char minValue = R;
		if (maxValue < G)
			maxValue = G;
		if (maxValue < B)
			maxValue = B;
		if (minValue > G)
			minValue = G;
		if (minValue > B)
			minValue = B;
		if (maxValue - minValue > thre_col) {
			pBinBuf[(start - pStart) / 3] = 255;
		}
		else {
			pBinBuf[(start - pStart) / 3] = 0;
		}
	}
	binImg = cv::Mat(img.rows, img.cols, CV_8UC1, pBinBuf.get(), cv::Mat::AUTO_STEP).clone();
	//cv::imwrite("D:\\TEST_OUTPUT\\rnnPredict\\binImg_f.tif", binImg);
	//对binImg二值图进行操作
	remove_small_objects(binImg, thre_vol / pow(slideRatio, level));
}

bool Model1Holder::initialize_binImg(MultiImageRead &mImgRead)
{
	int heightL4 = 0;
	int widthL4 = 0;
	mImgRead.getLevelDimensions(levelBin, widthL4, heightL4);
	if (widthL4 == 0 || heightL4 == 0) {
		cout << "get L4 image failed\n";
		return false;
	}
	cv::Mat imgL4;
	mImgRead.getTile(levelBin, 0, 0, widthL4, heightL4, imgL4);
	threshold_segmentation(imgL4, binImg, levelBin, m_thre_col, m_thre_vol);
	return true;
}

void Model1Holder::enterModel1Queue4(std::atomic<bool>& flag, MultiImageRead& mImgRead)
{
	flag = true;
	vector<std::pair<cv::Rect, cv::Mat>> tempRectMats;
	int cropSize = model1Height * float(model1Mpp / slideMpp);
	cropSize = cropSize / std::pow(slideRatio, levelBin);
	while (mImgRead.popQueue(tempRectMats)) {
		vector<std::pair<cv::Rect, cv::Mat>> rectMats;
		for (auto iter = tempRectMats.begin(); iter != tempRectMats.end(); iter++) {
			//cv::imwrite("D:\\TEST_OUTPUT\\rnnPredict\\" + to_string(iter->first.x) + "_" + to_string(iter->first.y) + ".tif", iter->second);
			bool flag_right = false;
			bool flag_down = false;
			if (iter->second.cols != block_width / std::pow(slideRatio, read_level))
				flag_right = true;
			if (iter->second.rows != block_height / std::pow(slideRatio, read_level))
				flag_down = true;
			int crop_width = int(model1Height * float(model1Mpp / slideMpp)) / std::pow(slideRatio, read_level);
			int crop_height = int(model1Width * float(model1Mpp / slideMpp)) / std::pow(slideRatio, read_level);
			int overlap = (int(model1Height * float(model1Mpp / slideMpp)) / 4) / std::pow(slideRatio, read_level);
			vector<cv::Rect> rects = iniRects(
				crop_width, crop_height,
				iter->second.rows, iter->second.cols, overlap, flag_right, flag_down);

			for (auto iter2 = rects.begin(); iter2 != rects.end(); iter2++) {
				std::pair<cv::Rect, cv::Mat> rectMat;
				cv::Rect rect;
				rect.x = iter->first.x + iter2->x;
				rect.y = iter->first.y + iter2->y;
				//这里过滤掉在binImg中和为0的图像
				int startX = rect.x / std::pow(slideRatio, levelBin - read_level);
				int startY = rect.y / std::pow(slideRatio, levelBin - read_level);
				cv::Rect rectCrop(startX, startY, cropSize, cropSize);
				cv::Mat cropMat = binImg(rectCrop);
				int cropSum = cv::sum(cropMat)[0];
				if (cropSum <= m_crop_sum * 255)
					continue;
				rect.width = model1Width;
				rect.height = model1Height;
				rectMat.first = rect;
				rectMat.second = iter->second(*iter2);
				cv::resize(rectMat.second, rectMat.second, Size(model1Height, model1Width));
				rectMats.emplace_back(std::move(rectMat));
			}
		}
		//将rectMats放到tensor的队列里面
		//vector<std::pair<vector<cv::Rect>, Tensor>> rectsTensors;
		//Mats2Tensors(rectMats, rectsTensors, model1_batchsize);
		//std::unique_lock<std::mutex> m1Guard(queue_lock);
		//for (auto iter = rectsTensors.begin(); iter != rectsTensors.end(); iter++) {
		//	data_queue.emplace(std::move(*iter));
		//}
		std::unique_lock<std::mutex> m1Guard(queue_lock);
		for (auto iter = rectMats.begin(); iter != rectMats.end(); iter++)
		{
			data_queue2.emplace(std::move(*iter));
		}
		m1Guard.unlock();
		queue_cv.notify_one();
		tempRectMats.clear();
	}
	flag = false;
}

vector<cv::Rect> Model1Holder::iniRects(int sHeight, int sWidth, int height, int width, int overlap, bool flag_right, bool flag_down)
{
	vector<cv::Rect> rects;
	//进行参数检查
	if (sHeight == 0 || sWidth == 0 || height == 0 || width == 0) {
		cout << "iniRects: parameter should not be zero\n";
		return rects;
	}
	if (sHeight > height || sWidth > width) {
		cout << "iniRects: sHeight or sWidth > height or width\n";
		return rects;
	}
	if (overlap >= sWidth || overlap >= height) {
		cout << "overlap should < sWidth or sHeight\n";
		return rects;
	}
	int x_num = (width - overlap) / (sWidth - overlap);
	int y_num = (height - overlap) / (sHeight - overlap);
	vector<int> xStart;
	vector<int> yStart;
	if ((x_num * (sWidth - overlap) + overlap) == width) {
		flag_right = false;
	}
	if ((y_num * (sHeight - overlap) + overlap) == height) {
		flag_down = false;
	}
	for (int i = 0; i < x_num; i++) {
		xStart.emplace_back((sWidth - overlap) * i);
	}
	for (int i = 0; i < y_num; i++) {
		yStart.emplace_back((sHeight - overlap) * i);
	}
	if (flag_right)
		xStart.emplace_back(width - sWidth);
	if (flag_down)
		yStart.emplace_back(height - sHeight);
	for (int i = 0; i < yStart.size(); i++) {
		for (int j = 0; j < xStart.size(); j++) {
			cv::Rect rect;
			rect.x = xStart[j];
			rect.y = yStart[i];
			rect.width = sWidth;
			rect.height = sHeight;
			rects.emplace_back(rect);
		}
	}
	return rects;
}

bool Model1Holder::checkFlags() {
	if (enterFlag3.load() || enterFlag4.load() || enterFlag5.load() || enterFlag6.load())
		return true;
	return false;
}

bool Model1Holder::popModel1Queue(vector<std::pair<cv::Rect, cv::Mat>> &rectMats/*vector<std::pair<vector<cv::Rect>, Tensor>>& rectsTensors*/)
{
	std::unique_lock < std::mutex > m1Guard(queue_lock);
	if (data_queue2.size() > 0) {
		int size = data_queue2.size();
		for (int i = 0; i < size; i++) {
			rectMats.emplace_back(std::move(data_queue2.front()));
			data_queue2.pop();
		}
		return true;
	}
	else if (checkFlags()) {
		queue_cv.wait_for(m1Guard, 3000ms, [this] {
			if (data_queue2.size() > 0)
				return true;
			return false;
			});
		int size = data_queue2.size();
		if (size == 0 && checkFlags()) {
			m1Guard.unlock();
			bool flag = popModel1Queue(rectMats);
			return flag;
		}
		if (size == 0 && !checkFlags()) {
			return false;
		}
		for (int i = 0; i < size; i++) {
			rectMats.emplace_back(std::move(data_queue2.front()));
			data_queue2.pop();
		}
		return true;
	}
	else
	{
		return false;
	}
}

vector<cv::Rect> Model1Holder::get_rects_slide()
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

vector<regionResult> Model1Holder::runModel1(MultiImageRead& mImgRead)
{
	//vector<cv::Rect> rects = iniRects(mImgRead);
	//vector<cv::Rect> rects = iniRects(model1Height * float(model1Mpp / slideMpp), slideWidth, slideHeight, slideWidth);
	//vector<cv::Rect> rects = iniRects(block_height, block_width, slideHeight, slideWidth, 560);
	//vector<cv::Rect> rects = get_rects_slide();

	iniPara(mImgRead);
	initialize_binImg(mImgRead);
	vector<regionResult> rResults;

	mImgRead.setReadLevel(read_level);
	vector<cv::Rect> rects = get_rects_slide();
	mImgRead.setRects(rects);

	//在这里开启4个线程
	//int count = 0;
	std::thread thread1(&Model1Holder::enterModel1Queue4, this, std::ref(enterFlag3), std::ref(mImgRead));
	std::thread thread2(&Model1Holder::enterModel1Queue4, this, std::ref(enterFlag4), std::ref(mImgRead));
	std::thread thread3(&Model1Holder::enterModel1Queue4, this, std::ref(enterFlag5), std::ref(mImgRead));
	std::thread thread4(&Model1Holder::enterModel1Queue4, this, std::ref(enterFlag6), std::ref(mImgRead));
	while (!checkFlags()) {
		continue;
	}
	//std::vector<std::pair<vector<cv::Rect>, Tensor>> rectsTensors;
	std::vector<std::pair<cv::Rect, cv::Mat>> rectMats;
	while (popModel1Queue(rectMats))
	{
		//vector<Tensor> tensors;
		//vector<cv::Rect> tmpRects;
		//for (auto iter = rectsTensors.begin(); iter != rectsTensors.end(); iter++) {
		//	tensors.emplace_back(std::move(iter->second));
		//	tmpRects.insert(tmpRects.end(), iter->first.begin(), iter->first.end());
		//}
		vector<cv::Mat> input_imgs;
		vector<cv::Rect> input_rects;
		for (auto iter = rectMats.begin(); iter != rectMats.end(); iter++)
		{
			input_rects.emplace_back(std::move(iter->first));
			input_imgs.emplace_back(std::move(iter->second));
		}
		//vector<model1Result> results = model1Handle->model1Process(input_imgs);
		model1Handle->processDataConcurrency(input_imgs);
		vector<model1Result> results = model1Handle->m_results;
		for (auto iter = results.begin(); iter != results.end(); iter++) {
			int place = iter - results.begin();
			regionResult rResult;
			rResult.result = *iter;
			rResult.point.x = input_rects[place].x * std::pow(slideRatio, read_level);//转为第0层级的图像
			rResult.point.y = input_rects[place].y * std::pow(slideRatio, read_level);
			rResults.emplace_back(rResult);
		}
		//count = count + rectsTensors.size();
		rectMats.clear();
	}
	cout << endl;
	thread1.join();
	thread2.join();
	thread3.join();
	thread4.join();

	return rResults;
	//int count = 0;
	//std::thread threadEnterQueue(&SlideProc::enterModel1Queue2, this, std::ref(mImgRead));
	//while (enterFlag1 != true)
	//{
	//	continue;
	//}
	//cout << endl;
	//while (/*mImgRead.popQueue(rectMats)*/popModel1Queue(rectMats))
	//{
	//	cout << count << " ";
	//	vector<cv::Mat> imgs;
	//	vector<cv::Rect> tmpRects;
	//	int size = rectMats.size();
	//	for (auto iter = rectMats.begin(); iter != rectMats.end(); iter++)
	//	{
	//		imgs.emplace_back(std::move(iter->second));//用move语义将其放到新的里面
	//		tmpRects.emplace_back(std::move(iter->first));
	//	}
	//	vector<model1Result> results = model1Handle->model1Process(imgs);
	//	//将得到的结果和全局的坐标信息放到rResults里面
	//	for (auto iter = results.begin(); iter != results.end(); iter++)
	//	{
	//		int place = iter - results.begin();
	//		regionResult rResult;
	//		rResult.result = *iter;
	//		rResult.point.x = tmpRects[place].x;
	//		rResult.point.y = tmpRects[place].y;
	//		rResults.emplace_back(rResult);
	//	}
	//	count = count + rectMats.size();
	//	rectMats.clear();		
	//}
	//threadEnterQueue.join();
}