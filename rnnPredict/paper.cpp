#include <iostream>
#include <fstream>
#include <filesystem>
#include "IniConfig.h"
#include "Model2Holder.h"
#include "RnnHolder.h"
#include "commonFunction.h"

bool get_high_level_img(MultiImageRead& mImgRead, cv::Mat &highLevelImg)
{
	double slideMpp = 0;
	mImgRead.getSlideMpp(slideMpp);
	int slideRatio = 0;
	slideRatio = mImgRead.get_ratio();
	int levelBin = 0;
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

	int heightL4 = 0;
	int widthL4 = 0;
	mImgRead.getLevelDimensions(levelBin, widthL4, heightL4);

	if (widthL4 == 0 || heightL4 == 0) {
		cout << "get L4 image failed\n";
		return false;
	}
	mImgRead.getTile(levelBin, 0, 0, widthL4, heightL4, highLevelImg);
	return true;
}

std::string getFileNamePrefix2(std::string path)
{
	std::filesystem::path fs_path(path);
	std::string filename = fs_path.filename().string();
	std::vector<std::string> split_str = split(filename, '.');
	std::string prefix = split_str[0];
	return prefix;
}

std::vector<PointScore> getRnnScorePoint(std::string txtPath)
{
	std::vector<PointScore> ret;
	std::vector<std::string> strs;
	std::ifstream file(txtPath);
	std::string str;
	int count = 0;
	while (std::getline(file, str)) {
		if (count >= 30) {
			break;
		}
		strs.emplace_back(str);
		count++;
	}
	for (auto elem : strs)
	{
		PointScore ps;
		std::vector<std::string> splitStr = split(elem, ',');
		ps.score = std::stod(splitStr[0]);
		ps.point.x = std::stoi(splitStr[1]);
		ps.point.y = std::stoi(splitStr[2]);
		ret.emplace_back(ps);
	}
	return ret;
}

float getRnnResult(Model2Holder * m2Holder, RnnHolder *rnnHolder, std::string slidePath, std::vector<PointScore>& pss)
{
	std::string filename = getFileNamePrefix2(slidePath);
	MultiImageRead mImgRead(slidePath.c_str());
	mImgRead.setGammaFlag(false);
	mImgRead.createThreadPool();
	double slideMpp;
	mImgRead.getSlideMpp(slideMpp);
	int model2Height = IniConfig::instance().getIniInt("Model2", "height");
	int model2Width = IniConfig::instance().getIniInt("Model2", "width");
	double model2Mpp = IniConfig::instance().getIniDouble("Model2", "mpp");
	std::vector<cv::Rect> rects;
	int crop_size = model2Height * 1.5;
	int crop_size2 = 1000;
	for (auto elem : pss)
	{
		//std::cout << elem.point.x << elem.point.y << std::endl;
		cv::Rect rect;
		rect.height = float(crop_size) * model2Mpp / slideMpp;
		rect.width = rect.height;
		rect.x = elem.point.x - int(float(rect.width) / 2.0f);
		rect.y = elem.point.y - int(float(rect.height) / 2.0f);
		rects.emplace_back(rect);
	}

	std::vector<cv::Rect> rects2;
	for (auto elem : pss)
	{
		cv::Rect rect;
		rect.height = float(crop_size2) * model2Mpp / slideMpp;
		rect.width = rect.height;
		rect.x = elem.point.x - int(float(rect.width) / 2.0f);
		rect.y = elem.point.y - int(float(rect.height) / 2.0f);
		rects2.emplace_back(rect);
	}
	if (rects2.size() > 10) {
		rects2.erase(rects2.begin() + 10, rects2.end());
	}

	std::string savePath = IniConfig::instance().getIniString("Config", "savePath");
	std::string trueSavePath = savePath + "\\" + filename;
	std::filesystem::create_directories(trueSavePath);
	vector<cv::Mat> imgs2;
	m2Holder->readImageInOrder(rects2, mImgRead, imgs2);
	int j = 0;
	for (auto& elem : imgs2)
	{
		cv::imwrite(trueSavePath + "\\" + std::to_string(j) + "_" + std::to_string(pss[j].point.x) +
			"_" + std::to_string(pss[j].point.y) + "_" + std::to_string(pss[j].score) + ".jpg", elem);
		//cv::imwrite(trueSavePath + "\\" + std::to_string(j) + ".tif", elem);
		j++;
	}
	
	vector<cv::Mat> imgs;
	m2Holder->readImageInOrder(rects, mImgRead, imgs);

	int i = 0;
	//std::string savePath = "G:\\manuRNN\\true_input\\";


	for (auto& elem : imgs)
	{
		//cv::imwrite(trueSavePath + "\\" + std::to_string(i) + ".jpg", elem);
		//cv::imwrite(savePath + std::to_string(i) + "_raw.tif", elem);
		//cv::imwrite(savePath + std::to_string(i) + "_raw.jpg", elem);
		//using namespace cv;
		//cv::imdecode(elem, cv::IMREAD_COLOR);
		std::vector<uchar> buf;
		cv::imencode(".jpg", elem, buf);
		cv::Mat dst = cv::imdecode(buf, cv::IMREAD_COLOR);
		//cv::imwrite(savePath + std::to_string(i) + "_encode.tif", dst);


		cv::resize(dst, dst, cv::Size(crop_size, crop_size));
		//按照paper的逻辑，先resize到384，在裁剪256，在丢进去跑出结果
		dst = dst(cv::Rect((crop_size - model2Width) / 2, (crop_size - model2Height) / 2, model2Width, model2Height)).clone();
		elem = std::move(dst);
		//cv::imwrite(trueSavePath + "\\" + std::to_string(i) + ".tif", elem);
		i++;
	}

	vector<model2Result> tempResults;
	m2Holder->model2Process(imgs, tempResults);

	float retScore = rnnHolder->runRnn(tempResults);

	//这里保存一下缩略图
	cv::Mat highLevelImg;
	if (get_high_level_img(mImgRead, highLevelImg))
	{
		cv::imwrite(trueSavePath + "\\thumbnail.jpg", highLevelImg);
	}

	return retScore;
}

float getRnnResult(Model2Holder* m2Holder, RnnHolder* rnnHolder, std::string txtPath, std::string slidePath)
{
	auto pss = getRnnScorePoint(txtPath);
	//std::cout << "11" << std::endl;
	auto score = getRnnResult(m2Holder, rnnHolder, slidePath, pss);
	
	return score;
}

void getRnnResults(std::string iniPath)
{
	std::string test = IniConfig::instance().getIniString("Config2", "test");
	std::cout << "test:" << test << std::endl;

	std::string slideParentPath = IniConfig::instance().getIniString("Config", "slidePath");
	std::string txtParentPath = IniConfig::instance().getIniString("Config", "savePath");

	std::cout << "slideParentPath:" << slideParentPath << std::endl;
	std::cout << "txtParentPath:" << txtParentPath << std::endl;

	std::vector<std::string> slideList;
	std::vector<std::string> tempSlideList;
	std::vector<std::string> txtList;
	std::vector<std::string> rnnList;
	getFiles(slideParentPath, tempSlideList, "sdpc");
	slideList.insert(slideList.end(), tempSlideList.begin(), tempSlideList.end());
	getFiles(slideParentPath, tempSlideList, "srp");
	slideList.insert(slideList.end(), tempSlideList.begin(), tempSlideList.end());
	getFiles(slideParentPath, tempSlideList, "mrxs");
	slideList.insert(slideList.end(), tempSlideList.begin(), tempSlideList.end());
	getFiles(slideParentPath, tempSlideList, "svs");
	slideList.insert(slideList.end(), tempSlideList.begin(), tempSlideList.end());

	getFiles(txtParentPath, txtList, "txt");
	std::cout << txtParentPath << std::endl;
	std::cout << "txt list size:" << txtList.size() << std::endl;
	getFiles(txtParentPath, rnnList, "rnnscore");
	filterList(txtList, rnnList);

	Model2Holder* m2Holder = new Model2Holder(iniPath);
	m2Holder->createThreadPool();
	RnnHolder* rnnHolder = new RnnHolder(iniPath);

	for (auto txt : txtList)
	{
		std::cout << txt << std::endl;
		std::string txtName = getFileNamePrefix2(txt);
		//从slideList中进行筛选
		for (auto slide : slideList)
		{
			std::cout << slide << "is running rnn\n";
			std::string slideName = getFileNamePrefix2(slide);
			if (txtName == slideName)
			{
				auto score = getRnnResult(m2Holder, rnnHolder, txt, slide);
				std::ofstream out(txtParentPath + "\\" + slideName + ".rnnscore");
				string saveString = std::to_string(score);
				out << saveString;
				out.close();
				break;
			}
			
		}
	}

	//std::vector<std::string> rnnScoreList;
	//std::vector<std::string> txtList2;
	//for (auto elem : txtList)
	//{
	//	if (elem.find("rnnScore") != std::string::npos) {
	//		rnnScoreList.emplace_back(elem);
	//	}
	//	else {
	//		txtList2.emplace_back(elem);
	//	}
	//}
	////根据rnnScore去掉已经计算过的

	//txtList = txtList2;
	////filterList(slideList, txtList);

	delete m2Holder;
	delete rnnHolder;
}

int main(int argc, char*argv[])
{
	//_putenv_s("CUDA_VISIBLE_DEVICES", "0");
	std::string iniPath = "";
	if (argc == 2) 
	{
		iniPath = std::string(argv[1]);
	}
	else if (argc == 1) {
		//iniPath = "./config.ini";
		iniPath = "./M12andRNNTest.ini";
	}
	else {
		return -1;
	}
	setIniPath(iniPath);
	std::cout << IniConfig::instance().getIniInt("Model2", "height") << std::endl;
	getRnnResults(iniPath);

	return 0;
}