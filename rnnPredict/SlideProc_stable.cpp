#include "SlideProc.h"
#include "tinyxml.h"
#include "tinystr.h"


void SlideProc::xgConfig(string xgParentPath)
{
	vector<string> xgPaths;
	getFiles(xgParentPath, xgPaths, "model");
	if (xgPaths.size() != 10) {
		cout << "xgboost model number should be 10\n";
		return;
	}
	for (auto iter = xgPaths.begin(); iter != xgPaths.end(); iter++) {
		int place = iter - xgPaths.begin();
		typedef handle(*function)(string);
		HINSTANCE xgDll = LoadLibraryA("xgdll.dll");
		function initialize_xgboost = nullptr;
		if (xgDll != NULL) {
			initialize_xgboost = (function)GetProcAddress(xgDll, "initialize_xgboost");
			if (initialize_xgboost != NULL) {
				//xgHandle[place] = initialize_xgboost(*iter);
				xgHandle.emplace_back(initialize_xgboost(*iter));
			}
			else {
				cout << "initialize_xgboost is null" << endl;
			}
		}
		else {
			cout << "cannot get xgdll, please check xgdll exist or not\n";
			return;
		}
	}
}

void SlideProc::rnnConfig(string rnnParentPath)
{
	vector<string> rnnPaths;
	getFiles(rnnParentPath, rnnPaths, "pb");
	if (rnnPaths.size() != 6) {
		cout << "rnn model number should be 6\n";
		return;
	}
	for (auto iter = rnnPaths.begin(); iter != rnnPaths.end(); iter++)
	{
		int place = iter - rnnPaths.begin();
		//读取模型
		modelConfig conf;
		conf.height = 256;//这些配置都无所谓了
		conf.width = 256;
		conf.channel = 3;
		conf.opsInput = "feature_input:0";
		conf.opsOutput.emplace_back("output/Sigmoid:0");
		std::ifstream file(*iter, std::ios::binary | std::ios::ate);
		std::streamsize size = file.tellg();
		char* buffer = new char[size];
		file.seekg(0, std::ios::beg);
		if (!file.read(buffer, size)) {
			cout << "read file to buffer failed" << endl;
		}
		rnn* rnnBase = new rnn(conf, buffer, size);
		rnnHandle.emplace_back(rnnBase);
		delete[]buffer;
	}
}

void SlideProc::model1Config(string model1Path)
{
	//modelConfig conf;
	//conf.height = 512;
	//conf.width = 512;
	//conf.channel = 3;
	//conf.opsInput = "input_1:0";
	//conf.opsOutput.emplace_back("dense_2/Sigmoid:0");
	//conf.opsOutput.emplace_back("conv2d_1/truediv:0");

	//std::ifstream file(model1Path, std::ios::binary | std::ios::ate);
	//std::streamsize size = file.tellg();
	////char* buffer = new char[size];
	//std::unique_ptr<char[]> uBuffer(new char[size]);
	//file.seekg(0, std::ios::beg);
	//if (!file.read(uBuffer.get(), size)) {
	//	std::cout << "read file to buffer failed" << endl;
	//}
	//model1Handle = new model1(conf, uBuffer.get(), size);
	model1Mpp = 0.586f;
	model1Height = 512;
	model1Width = 512;
}

void SlideProc::model2Config(string model2Path)
{
	//modelConfig conf;
	//conf.height = 256;
	//conf.width = 256;
	//conf.channel = 3;
	//conf.opsInput = "input_1:0";
	//conf.opsOutput.emplace_back("dense_2/Sigmoid:0");
	//conf.opsOutput.emplace_back("global_max_pooling2d_1/Max:0");

	//std::ifstream file(model2Path, std::ios::binary | std::ios::ate);
	//std::streamsize size = file.tellg();
	//std::unique_ptr<char[]> uBuffer(new char[size]);
	//file.seekg(0, std::ios::beg);
	//if (!file.read(uBuffer.get(), size)) {
	//	std::cout << "read file to buffer failed" << endl;
	//}
	//model2Handle = new model2(conf, uBuffer.get(), size);
	model2Mpp = 0.293f;
	model2Height = 256;
	model2Width = 256;
}

void SlideProc::model3Config(string model3Path)
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
}

void SlideProc::initialize_handler(const char* iniPath)
{
	//读取ini文件中的相关信息
	char model1Path[MAX_PATH];
	char model2Path[MAX_PATH];
	char model3Path[MAX_PATH];
	char rnnParentPath[MAX_PATH];
	//char xgParentPath[MAX_PATH];
	char group[] = "Config";
	char pbFile_n_1[] = "model1Path";
	char pbFile_n_2[] = "model2Path";
	char pbFile_n_3[] = "model3Path";
	char pbFile_n_6[] = "rnnParentPath";
	//char pbFile_n_7[] = "xgParentPath";

	GetPrivateProfileString(group, pbFile_n_1, "default", model1Path, MAX_PATH, iniPath);
	GetPrivateProfileString(group, pbFile_n_2, "default", model2Path, MAX_PATH, iniPath);
	GetPrivateProfileString(group, pbFile_n_3, "default", model3Path, MAX_PATH, iniPath);
	GetPrivateProfileString(group, pbFile_n_6, "default", rnnParentPath, MAX_PATH, iniPath);
	//GetPrivateProfileString(group, pbFile_n_7, "default", xgParentPath, MAX_PATH, iniPath);

	model1Config(string(model1Path));
	m1Holder = new Model1Holder(model1Path);
	model2Config(string(model2Path));
	m2Holder = new Model2Holder(model2Path);
	model3Config(string(model3Path));
	rnnConfig(string(rnnParentPath));
	//xgConfig(string(xgParentPath));
}

void SlideProc::remove_small_objects(cv::Mat& binImg, int thre_vol)
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

void SlideProc::threshold_segmentation(cv::Mat& img, cv::Mat& binImg, int level, int thre_col, int thre_vol)
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


void SlideProc::saveResult(string savePath, string filename)
{
	TiXmlDocument* writeDoc = new TiXmlDocument;//xml文档指针	
	TiXmlDeclaration* decl = new TiXmlDeclaration("1.0", "UTF-8", "yes");//文档格式声明
	writeDoc->LinkEndChild(decl);//写入文档

	TiXmlElement* RootElement = new TiXmlElement("Slide");
	RootElement->SetAttribute("name", filename.c_str());
	RootElement->SetAttribute("num", rResults.size());
	writeDoc->LinkEndChild(RootElement);

	//int count = 0;
	for (auto& elem : rResults) {
		TiXmlElement* BlockElem = new TiXmlElement("Block");
		RootElement->LinkEndChild(BlockElem);
		BlockElem->SetAttribute("X", elem.point.x);
		BlockElem->SetAttribute("Y", elem.point.y);
		BlockElem->SetAttribute("score", to_string(elem.result.score).c_str());
		bool model2Flag = false;
		if (elem.score2.size() > 0)
			model2Flag = true;
		for (auto iter = elem.result.points.begin() + 1; iter != elem.result.points.end(); iter++) {
			int place = iter - elem.result.points.begin() - 1;
			TiXmlElement* LocateElem = new TiXmlElement("Locate");
			BlockElem->LinkEndChild(LocateElem);
			if (model2Flag) {
				LocateElem->SetAttribute("score", to_string(elem.score2[place]).c_str());
			}
			else {
				LocateElem->SetAttribute("score", "-1");
			}
			cv::Point point;
			point.x = elem.point.x + iter->x * float(model1Mpp / slideMpp);
			point.y = elem.point.y + iter->y * float(model1Mpp / slideMpp);
			//cv::Mat img;
			int saveHeight = 256 * float(model2Mpp / slideMpp);
			//m_sdpcRead->getTile(0, point.x - saveHeight / 2, point.y - saveHeight / 2, saveHeight, saveHeight, img);
			//cv::imwrite("D:\\TEST_OUTPUT\\rnnPredict\\" + to_string(count) + ".tif", img);
			//count++;
			LocateElem->SetAttribute("X", point.x);
			LocateElem->SetAttribute("Y", point.y);
		}
	}
	writeDoc->SaveFile((savePath + "\\" + filename + ".xml").c_str());
	delete writeDoc;
}
