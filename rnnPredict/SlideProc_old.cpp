#include "SlideProc.h"

vector<cv::Rect> SlideProc::iniRects(int sHeight, int sWidth, int height, int width, int overlap)
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
	bool flag_right = true;
	bool flag_down = true;
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
		xStart.emplace_back((sWidth - overlap) * x_num);
	if (flag_down)
		yStart.emplace_back((sHeight - overlap) * y_num);
	int last_width = width - x_num * (sWidth - overlap);
	int last_height = height - y_num * (sHeight - overlap);
	for (int i = 0; i < yStart.size(); i++) {
		for (int j = 0; j < xStart.size(); j++) {
			cv::Rect rect;
			rect.x = xStart[j];
			rect.y = yStart[i];
			rect.width = sWidth;
			rect.height = sHeight;
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

vector<cv::Rect> SlideProc::iniRects(int sHeight, int sWidth, int height, int width)
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
	int xNum = 0;//水平方向的裁剪个数
	int yNum = 0;//垂直方向的裁剪个数
	int overlap = sHeight * (0.25f);
	if (sHeight <= overlap || sWidth <= overlap) {
		cout << "sHeight and sWidth seems to small\n";
		return rects;
	}
	yNum = 1 + (height - sHeight) / (sHeight - overlap);
	xNum = 1 + (width - sWidth) / (sWidth - overlap);
	vector<int> xStart;
	vector<int> yStart;
	for (int i = 0; i < xNum; i++) {
		xStart.emplace_back((sWidth - overlap) * i);
	}
	for (int i = 0; i < yNum; i++) {
		yStart.emplace_back((sHeight - overlap) * i);
	}
	int xLeft = width - xNum * sWidth;
	int yLeft = height - yNum * sHeight;
	if (xLeft != 0)
		xStart.emplace_back(width - sWidth);
	if (yLeft != 0)
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

//vector<cv::Rect> SlideProc::iniRects(MultiImageRead& mImgRead)
//{
//	int sHeight = model1Handle->getModelHeight();
//	int sWidth = model1Handle->getModelWidth();
//	double mpp = 0.0f;
//	mImgRead.getSlideMpp(mpp);
//	sHeight = sHeight * float(model1Handle->getM1Resolution() / mpp);
//	sWidth = sWidth * float(model1Handle->getM1Resolution() / mpp);
//	int height = 0;
//	mImgRead.getSlideHeight(height);
//	int width = 0;
//	mImgRead.getSlideWidth(width);
//	vector<cv::Rect> rects;
//	if (height == 0 || width == 0 || sHeight == 0 || sWidth == 0) {
//		cout << "iniRects: some parameters should not be zero\n";
//		return rects;
//	}
//	if (height < sHeight || width < sWidth) {
//		cout << "size to be cropped should bigger \n";
//		return rects;
//	}
//	int xNum = 0;//水平方向的裁剪个数
//	int yNum = 0;//垂直方向的裁剪个数
//	int overlap = sHeight * (0.25f);
//	if (sHeight <= overlap || sWidth <= overlap) {
//		cout << "sHeight and sWidth seems to small\n";
//		return rects;
//	}
//	yNum = 1 + (height - sHeight) / (sHeight - overlap);
//	xNum = 1 + (width - sWidth) / (sWidth - overlap);
//	vector<int> xStart;
//	vector<int> yStart;
//	for (int i = 0; i < xNum; i++) {
//		xStart.emplace_back((sWidth - overlap) * i);
//	}
//	for (int i = 0; i < yNum; i++) {
//		yStart.emplace_back((sHeight - overlap) * i);
//	}
//	int xLeft = width - xNum * sWidth;
//	int yLeft = height - yNum * sHeight;
//	if (xLeft != 0)
//		xStart.emplace_back(width - sWidth - 1);
//	if (yLeft != 0)
//		yStart.emplace_back(height - sHeight - 1);
//	for (int i = 0; i < yStart.size(); i++) {
//		for (int j = 0; j < xStart.size(); j++) {
//			cv::Rect rect;
//			rect.x = xStart[j];
//			rect.y = yStart[i];
//			rect.width = sWidth;
//			rect.height = sHeight;
//			rects.emplace_back(rect);
//		}
//	}
//	return rects;
//}

vector<cv::Rect> SlideProc::get_rects_slide()
{
	vector<cv::Rect> rects;
	int overlap = 560;
	//开始计算overlap
	int x_num = (slideWidth - overlap) / (block_width - overlap);
	int y_num = (slideHeight - overlap) / (block_height - overlap);
	int sHeight = model1Height * float(model1Mpp / slideMpp);
	int sWidth = model1Width * float(model1Mpp / slideMpp);
	vector<int> xStart;
	vector<int> yStart;
	bool flag_right = true;
	bool flag_down = true;
	if ((x_num * (block_width - overlap) + overlap) == slideWidth) {
		flag_right = false;
	}
	if ((y_num * (block_height - overlap) + overlap) == slideHeight) {
		flag_down = false;
	}
	for (int i = 0; i < x_num; i++) {
		xStart.emplace_back((block_width - overlap) * i);
	}
	for (int i = 0; i < y_num; i++) {
		yStart.emplace_back((block_height - overlap) * i);
	}
	int last_width = slideWidth - x_num * (block_width - overlap);
	int last_height = slideHeight - y_num * (block_height - overlap);
	if (flag_right) {
		if (last_width >= sWidth)
			xStart.emplace_back((block_width - overlap) * x_num);
		else {
			xStart.emplace_back(slideWidth - sWidth);
			last_width = sWidth;
		}
	}
	if (flag_down) {
		if (last_height >= sHeight)
			yStart.emplace_back((block_height - overlap) * y_num);
		else {
			yStart.emplace_back(slideHeight - sHeight);
			last_height = sHeight;
		}
	}
	for (int i = 0; i < yStart.size(); i++) {
		for (int j = 0; j < xStart.size(); j++) {
			cv::Rect rect;
			rect.x = xStart[j];
			rect.y = yStart[i];
			rect.width = block_width;
			rect.height = block_height;
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

vector<cv::Rect> SlideProc::iniRects(int sHeight, int sWidth, int height, int width, int overlap, bool flag_right, bool flag_down)
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


bool SlideProc::popModel1Queue(vector<std::pair<cv::Rect, cv::Mat>>& rectMats)
{
	//先加锁
	std::unique_lock < std::mutex > m1Guard(queue1Lock);
	if (model1Queue.size() > 0) {
		int size = model1Queue.size();
		for (int i = 0; i < size; i++) {
			rectMats.emplace_back(std::move(model1Queue.front()));
			model1Queue.pop();
		}
		return true;
	}
	else if (enterFlag1.load()) {
		//如果这个enterQueue1线程没有退出，那么就开始进行wait操作
		queue_cv1.wait_for(m1Guard, 3000ms, [this] {
			if (model1Queue.size() > 0)
				return true;
			return false;
			});
		//取到锁之后，再次进行循环
		int size = model1Queue.size();
		if (size == 0 && enterFlag1.load()) {
			m1Guard.unlock();
			bool flag = popModel1Queue(rectMats);
			return flag;
		}
		if (size == 0 && !enterFlag1.load()) {
			return false;
		}
		for (int i = 0; i < size; i++) {
			rectMats.emplace_back(std::move(model1Queue.front()));
			model1Queue.pop();
		}
		return true;
	}
	else {
		return false;
	}
}

void SlideProc::enterModel1Queue2(MultiImageRead& mImgRead)
{
	enterFlag1 = true;
	vector<std::pair<cv::Rect, cv::Mat>> tempRectMats;
	int cropSize = model1Height * float(model1Mpp / slideMpp);
	cropSize = cropSize / std::pow(slideRatio, levelBin);
	while (mImgRead.popQueue(tempRectMats)) {
		//对tempRectMats进行裁剪，然后resize之后送入model1Queue
		vector<std::pair<cv::Rect, cv::Mat>> rectMats;
		for (auto iter = tempRectMats.begin(); iter != tempRectMats.end(); iter++) {
			vector<cv::Rect> rects = iniRects(model1Height * float(model1Mpp / slideMpp), model1Width * float(model1Mpp / slideMpp), iter->second.rows, iter->second.cols);
			for (auto iter2 = rects.begin(); iter2 != rects.end(); iter2++) {
				std::pair<cv::Rect, cv::Mat> rectMat;
				cv::Rect rect;
				rect.x = iter->first.x + iter2->x;
				rect.y = iter->first.y + iter2->y;
				//这里过滤掉在binImg中和为0的图像
				int startX = rect.x / std::pow(slideRatio, levelBin);
				int startY = rect.y / std::pow(slideRatio, levelBin);
				cv::Rect rectCrop(startX, startY, cropSize, cropSize);
				cv::Mat cropMat = binImg(rectCrop);
				int cropSum = cv::sum(cropMat)[0];
				if (cropSum == 0)
					continue;
				rect.width = model1Width;
				rect.height = model1Height;
				rectMat.first = rect;
				rectMat.second = iter->second(*iter2);
				cv::resize(rectMat.second, rectMat.second, cv::Size(model1Height, model1Width));
				rectMats.emplace_back(std::move(rectMat));
			}
		}
		std::unique_lock<std::mutex> m1Guard(queue1Lock);
		for (auto iter = rectMats.begin(); iter != rectMats.end(); iter++) {
			model1Queue.emplace(std::move(*iter));
		}
		m1Guard.unlock();
		queue_cv1.notify_one();
		tempRectMats.clear();
	}
	enterFlag1 = false;
}

void SlideProc::enterModel1Queue(MultiImageRead& mImgRead)
{
	enterFlag1 = true;
	vector<std::pair<cv::Rect, cv::Mat>> tempRectMats;
	while (mImgRead.popQueue(tempRectMats)) {
		for (auto iter = tempRectMats.begin(); iter != tempRectMats.end(); iter++) {
			if (iter->second.cols != model1Width) {
				cv::resize(iter->second, iter->second, cv::Size(model1Width, model1Height));
			}
		}
		std::unique_lock<std::mutex> m1Guard(queue1Lock);
		for (auto iter = tempRectMats.begin(); iter != tempRectMats.end(); iter++) {
			model1Queue.emplace(std::move(*iter));
		}
		m1Guard.unlock();
		queue_cv1.notify_one();
		tempRectMats.clear();
	}
	enterFlag1 = false;
}


//bool SlideProc::runSlide(const char* slide)
//{
//	//初始化片子的信息
//	if (m_srpRead != nullptr) {
//		delete m_srpRead;
//		m_srpRead = new SrpSlideRead(slide);
//	}
//	else {
//		m_srpRead = new SrpSlideRead(slide);
//	}
//
//	initialize_binImg();
//
//	m_slide = string(slide);
//	rResults.clear();
//	MultiImageRead mImgRead(slide);
//	mImgRead.getSlideHeight(slideHeight);
//	mImgRead.getSlideWidth(slideWidth);
//	mImgRead.getSlideMpp(slideMpp);
//	slideRatio = mImgRead.get_ratio();
//	if (slideHeight == 0 || slideWidth == 0 || slideMpp == 0)
//		return false;
//	mImgRead.createThreadPool();
//	mImgRead.setAddTaskThread();
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
//	vector<Anno> annos = regionProposal(mImgRead);
//	now = time(0);
//	cout << "start runRnn: " << (char*)ctime(&now);
//	float rnnResult = runRnn(annos, mImgRead);
//	float xgResult = runXgboost();
//	sortResultsByCoor(rResults);
//	float theEndScore = 0.0f;
//	mImgRead.~MultiImageRead();//提前析构掉
//
//	if (rnnResult < 0.1f) {
//		theEndScore = (rnnResult + xgResult) / 2;
//	}
//	else {
//		theEndScore = rnnResult;
//	}
//	//将信息写到srp里面
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
//	m_srpRead->callWriteParamDouble("score", theEndScore);
//	slideScore = theEndScore;
//	return true;
//}
