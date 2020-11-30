#include "Model1Holder.h"
#include "progress_record.h"
Model1Holder::Model1Holder(string iniPath)
{
	if (IniConfig::instance().getIniString("TensorRT", "USE_TR") == "ON")
		use_tr = true;
	else if (IniConfig::instance().getIniString("TensorRT", "USE_TR") == "OFF")
		use_tr = false;
	else
		use_tr = false;
	model1Config(iniPath);
}

Model1Holder::~Model1Holder()
{
	stopped.store(true);
	data_cv.notify_all();
	task_cv.notify_all();
	for (std::thread& thread : pool) {
		if (thread.joinable())
			thread.join();
	}
}

void Model1Holder::model1Config(string iniPath)
{
	if (!use_tr)
	{
		model1Handle.first = std::make_unique<TfModel1>("TfModel1");
		model1Handle.first->createThreadPool();
	}
	else
	{
		model1Handle.second = std::make_unique<TrModel1>("TrModel1");
		model1Handle.second->createThreadPool();
	}
	model1Mpp = IniConfig::instance().getIniDouble("Model1", "mpp");
	model1Height = IniConfig::instance().getIniDouble("Model1", "height");
	model1Width = IniConfig::instance().getIniDouble("Model1", "width");
}

void Model1Holder::pushData(MultiImageRead& mImgRead)
{
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
			int temp_rows = 0;
			int temp_cols = 0;
			vector<cv::Rect> rects = iniRects(
				crop_width, crop_height,
				iter->second.rows, iter->second.cols, overlap, flag_right, flag_down,
				temp_rows, temp_cols);

			for (auto iter2 = rects.begin(); iter2 != rects.end(); iter2++) {
				std::pair<cv::Rect, cv::Mat> rectMat;
				cv::Rect rect;
				rect.x = iter->first.x + iter2->x;
				rect.y = iter->first.y + iter2->y;
				//这里过滤掉在binImg中和为0的图像
				int startX = rect.x / std::pow(slideRatio, levelBin - read_level);
				int startY = rect.y / std::pow(slideRatio, levelBin - read_level);
				cv::Rect rectCrop(startX, startY, cropSize, cropSize);
				if (startX + cropSize > binImg.cols || startY + cropSize > binImg.rows)
				{
					rect.width = model1Width;
					rect.height = model1Height;
					rectMat.first = rect;
					rectMat.second = iter->second(*iter2);
					rectMats.emplace_back(std::move(rectMat));
					continue;
				}
				cv::Mat cropMat = binImg(rectCrop);
				int cropSum = cv::sum(cropMat)[0];
				if (cropSum <= m_crop_sum * 255)
				{
					addStep(1);
					continue;
				}
				rect.width = model1Width;
				rect.height = model1Height;
				rectMat.first = rect;
				rectMat.second = iter->second(*iter2);
				rectMats.emplace_back(std::move(rectMat));
			}
		}
		MultiThreadQueue::pushData(rectMats);
		tempRectMats.clear();
	}
}


vector<regionResult> Model1Holder::runModel1(MultiImageRead& mImgRead)
{
	//std::call_once(create_thread_flag, &Model1Holder::createThreadPool, this, 3);
	createThreadPool(3);

	iniPara(mImgRead);
	initialize_binImg(mImgRead);
	vector<regionResult> rResults;

	mImgRead.setReadLevel(read_level);
	vector<cv::Rect> rects = get_rects_slide();
	mImgRead.setRects(rects);
	int temp_cols = 0;
	int temp_rows = 0;
	vector<cv::Rect> rects2 = iniRects(
		model1Height * float(model1Mpp / slideMpp),
		model1Width * float(model1Mpp / slideMpp),
		slideHeight,
		slideWidth,
		(0.25f * model1Height) * float(model1Mpp / slideMpp),
		true,
		true,
		temp_rows, 
		temp_cols
	);
	setGlobalSlideHeight(temp_rows);
	setGlobalSlideWidth(temp_cols);
	setStage(0, rects2.size());

	//按照totalThrNum来决定进入几个task
	for (int i = 0; i < totalThrNum; i++)
	{
		auto task = std::make_shared<std::packaged_task<void()>>
			(std::bind(&Model1Holder::pushData,this, std::ref(mImgRead)));
		std::unique_lock<std::mutex> task_lock(task_mutex);
		tasks.emplace(
			[task]() {
				(*task)();
			}
		);
		task_lock.unlock();
		task_cv.notify_one();
	}

	std::vector<std::pair<cv::Rect, cv::Mat>> rectMats;
	while (popData(rectMats))
	{
		vector<cv::Mat> input_imgs;
		vector<cv::Rect> input_rects;
		for (auto iter = rectMats.begin(); iter != rectMats.end(); iter++)
		{
			input_rects.emplace_back(std::move(iter->first));
			input_imgs.emplace_back(std::move(iter->second));
		}
		vector<model1Result> results;
		//model1Handle->processDataConcurrency(input_imgs);
		//vector<model1Result> results = model1Handle->m_results;
		if (!use_tr)
		{
			model1Handle.first->processDataConcurrency(input_imgs);
			results = model1Handle.first->m_results;
		}
		else
		{
			model1Handle.second->processDataConcurrency(input_imgs);
			results = model1Handle.second->m_results;
		}
		for (auto iter = results.begin(); iter != results.end(); iter++) {
			int place = iter - results.begin();
			regionResult rResult;
			rResult.result = *iter;
			rResult.point.x = input_rects[place].x * std::pow(slideRatio, read_level);//转为第0层级的图像
			rResult.point.y = input_rects[place].y * std::pow(slideRatio, read_level);
			rResults.emplace_back(rResult);
		}
		rectMats.clear();
	}
	return rResults;
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

/*
@param:
	sHeight   :  小块的高
	sWidth    :  小块的宽
	height    :  大块的高
	width     :  大块的宽
	overlap   :  小块的重叠
	flag_right:  当右边有剩余时是否多裁
	flag_down :  当下面有剩余时是否多裁
@return:
    vector<cv::Rect>:应该裁取的框框
*/
vector<cv::Rect> Model1Holder::iniRects(
	int sHeight, int sWidth, int height, int width,
	int overlap, bool flag_right, bool flag_down, 
	int &rows, int &cols)
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
	cols = xStart.size();
	rows = yStart.size();
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

void Model1Holder::remove_small_objects(cv::Mat& binImg, int thre_vol)
{
	//去除img中小的区域
	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierarchy;
	findContours(binImg, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	double threshold = thre_vol;//面积的阈值
	vector<vector<cv::Point>> finalContours;
	for (int i = 0; i < contours.size(); i++) {
		double area = cv::contourArea(contours[i]);
		if (area >= threshold) {
			finalContours.emplace_back(contours[i]);
		}
	}
	if (finalContours.size() > 0) {
		cv::Mat finalMat(binImg.rows, binImg.cols, CV_8UC1, cv::Scalar(0));
		cv::fillPoly(finalMat, finalContours, cv::Scalar(255));
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

bool Model1Holder::initialize_binImg(MultiImageRead& mImgRead)
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