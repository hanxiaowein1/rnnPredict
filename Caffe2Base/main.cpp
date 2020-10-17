/*
	just for test
*/

#include "DetectModel.h"
#include "CommonFunction.h"
#include "IniConfig.h"


int main()
{
	_putenv_s("CUDA_VISIBLE_DEVICES", "0");
	setIniPath("../x64/Release/config.ini");
	DetectModel detect_model("DetectModel");
	detect_model.createThreadPool();
	std::string cv_img_path = "D:\\TEST_DATA\\1100037_1\\12_3.tif";
	cv::Mat img = cv::imread(cv_img_path);
	std::vector<cv::Mat> imgs;
	imgs.emplace_back(img);
	detect_model.processDataConcurrency(imgs);
	auto result = detect_model.m_result;
	for (auto elem : result)
	{
		cout << elem.first << ": " << elem.second << endl;
	}
	system("pause");
	return 0;
}