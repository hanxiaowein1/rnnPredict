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
	//if (m_srpRead != nullptr) {
	//	delete m_srpRead;
	//	m_srpRead = new SrpSlideRead(slide);
	//
	//}
	//else {
	//	m_srpRead = new SrpSlideRead(slide);
	//}
	//if (!m_srpRead->status())
	//	return false;
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
	
	//将10张图像放到model2中进行预测得到tensor
	vector<cv::Mat> imgs;
	m2Holder->readImageInOrder(rects, mImgRead, imgs);
	vector<model2Result> tempResults;
	m2Holder->model2Process(imgs, tempResults);

	float retScore = rnnHolder->runRnn(tempResults);
	return retScore;
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
		cv::Point point = iter->point;
		int place = iter - psCopy.begin();
		bool flag = true;
		for (auto iter2 = PointScores.begin(); iter2 != PointScores.end(); iter2++) {
			cv::Point point2 = iter2->point;
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

bool SlideProc::checkFlags2()
{
	if (enterFlag7.load() || enterFlag8.load() || enterFlag9.load() || enterFlag10.load())
		return true;
	return false;
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

void SlideProc::saveImages(vector<PointScore>& pss, int radius, string savePath, MultiImageRead &mImgRead)
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
		mImgRead.getTile(0, point.x, point.y, radius * 2, radius * 2, img);
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
	now = time(0);
	cout << "start rnn process: " << (char*)ctime(&now);
	slideScore = runRnn(annos, mImgRead);

	if (IniConfig::instance().getIniString("Model3", "state") != "ON")
	{
		//如果没有开启model3，那么只保留model2的结果
		annos = regionProposal(10);
		string model2ResultPath = in_savePath + "\\model2";
		createDirRecursive(model2ResultPath);
		vector<PointScore> pss = anno2PS(annos);
		int saveRadius = 500;
		saveImages(pss, saveRadius, model2ResultPath, mImgRead);
		mImgRead.~MultiImageRead();
	}
	else
	{
		//在这里测试一下model3
		vector<Anno> annos_for_m3 = regionProposal(50);
		now = time(0);
		cout << "start model3 process: " << (char*)ctime(&now);
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
			if (m3Results.size() > 10)
				m3Results.erase(m3Results.begin() + 10, m3Results.end());
		}
		now = time(0);
		cout << "start save image: " << (char*)ctime(&now);
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
		int saveRadius = 500;
		saveImages(pss, saveRadius, model2ResultPath, mImgRead);
		annos.clear();
		annos = regionProposal(50);
		pss = anno2PS(annos);
		saveImages(pss, saveRadius, model3InputPath, mImgRead);//可能会越界
		saveImages(m3Results, saveRadius, model3ResultPath, mImgRead);//可能会越界
		mImgRead.~MultiImageRead();
	}

}


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

bool SlideProc::runSlide(const char* slide, vector<Anno>& annos, int len)
{
	//levelBin = 4;

	//初始化片子的信息
	MultiImageRead mImgRead(slide);
	mImgRead.createThreadPool();
	if (!mImgRead.status())
		return false;
	std::cout << "read handle initialized\n";
	iniPara(slide, mImgRead);
	std::cout << "get slide info done\n";
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

	annos = regionProposal(30);
	now = time(0);
	cout << "start rnn process: " << (char*)ctime(&now);
	slideScore = runRnn(annos, mImgRead);

	mImgRead.~MultiImageRead();

	annos = regionProposal(len);
	//将信息写到srp里面
	//annos.erase(annos.begin() + 10, annos.end());
	Anno* pann = new Anno[annos.size()];
	for (int i = 0; i < annos.size(); i++) {
		pann[i].id = i;
		pann[i].type = 0;
		pann[i].x = annos[i].x;
		pann[i].y = annos[i].y;
		pann[i].score = annos[i].score;
	}
	now = time(0);
	cout << "write anno to slide: " << (char*)ctime(&now);
	std::unique_ptr<SrpSlideRead> srp_read_handle = std::make_unique<SrpSlideRead>(slide);
	srp_read_handle->callCleanAnno();
	srp_read_handle->callBeginBatch();
	srp_read_handle->callWriteAnno(pann, annos.size());
	srp_read_handle->callEndBatch();
	srp_read_handle->callWriteParamDouble("score", slideScore);
	delete[]pann;
	return true;
}
