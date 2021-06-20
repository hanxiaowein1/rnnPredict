#include <iostream>
#include "TrModel2.h"
#include "IniConfig.h"
#include "CommonFunction.h"
//测试TensorRT的模型量化后的精度损失情况
int main()
{
	_putenv_s("CUDA_VISIBLE_DEVICES", "0");
	std::string iniPath = "./config.ini";
	setIniPath(iniPath);
	std::cout << IniConfig::instance().getIniString("TrModel2", "memory") << std::endl;;
	TrModel2 trModel2("TrModel2");
	trModel2.createThreadPool();
	std::string img_path = "D:\\Programmer\\Python\\测试代码\\pybind11_test\\image_save\\";
	std::vector<std::string> img_paths;
	getFiles(img_path, img_paths, "jpg");
	std::vector<cv::Mat> imgs;

	for (auto elem : img_paths) {
		cv::Mat img = cv::imread(elem);
		
		imgs.emplace_back(std::move(img));
	}

	trModel2.processDataConcurrency(imgs);

	auto result = trModel2.m_results;
	for (auto elem : result) {
		std::cout << elem.score << std::endl;
	}
	//std::cout << result[0].score << std::endl;
	system("pause");
	return 0;
}