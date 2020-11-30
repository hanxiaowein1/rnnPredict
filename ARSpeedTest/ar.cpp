#include "ar.h"
#include "IniConfig.h"
#include "TrModel1.h"
#include "TrModel2.h"
#include "commonFunction.h"
#include <iostream>
#include <filesystem>

extern vector<cv::Rect> iniRects(
	int sHeight, int sWidth, int height, int width,
	int overlap, bool flag_right, bool flag_down,
	int& rows, int& cols);

extern void getSubMat(cv::Mat& src, cv::Mat& dst, int center_x, int center_y, int height, int width);
extern void filterBaseOnPoint(vector<PointScore>& PointScores, int threshold);

struct TrModelHolder {
	TrModel1* tr_model1 = nullptr;
	TrModel2* tr_model2 = nullptr;
};

ArHandle initialize_handle(std::string ini_path)
{
	setIniPath(ini_path.c_str());
	TrModelHolder* tr_model_holder = new TrModelHolder;
	tr_model_holder->tr_model1 = new TrModel1("TrModel1");
	tr_model_holder->tr_model1->createThreadPool(1);
	tr_model_holder->tr_model2 = new TrModel2("TrModel2");
	tr_model_holder->tr_model2->createThreadPool(1);
	return ArHandle(tr_model_holder);
}

void process(std::vector<Anno>& annos, ArHandle myHandle, cv::Mat& raw_img, double img_mpp, int n)
{
	TrModelHolder* tr_model_holder = (TrModelHolder*)myHandle;
	if (!tr_model_holder)
	{
		assert("ArHandle initialize failed\n");
	}
	if (!tr_model_holder->tr_model1)
	{
		assert("Tensorrt model1 initialize failed\n");
	}
	if (!tr_model_holder->tr_model2)
	{
		assert("Tensorrt model2 initialize failed\n");
	}
	double model1_mpp = 0.586f;
	int    model1_width = 512;
	int    model1_height = 512;
	double model2_mpp = 0.293f;
	int    model2_width = 256;
	int    model2_height = 256;

	auto start = std::chrono::system_clock::now();


	//先获取model1的图像
	cv::Mat model1_img;
	if (img_mpp != 0.586f)
	{
		int new_height = raw_img.rows / float(model1_mpp / img_mpp);
		int new_width = raw_img.cols / float(model1_mpp / img_mpp);
		cv::resize(raw_img, model1_img, cv::Size(new_width, new_height));
	}
	else
		model1_img = raw_img;
	//如果model1_img的宽高并没有达到512，那么我就补，哪边不够补哪边
	if (model1_img.rows < 512 || model1_img.cols < 512)
	{
		//开始补
		int new_height = model1_img.rows < 512 ? 512 : model1_img.rows;
		int new_width = model1_img.cols < 512 ? 512 : model1_img.cols;
		cv::Mat white_mat(new_height, new_width, CV_8UC3, cv::Scalar(0, 0, 0));
		model1_img.copyTo(white_mat(cv::Rect(0, 0, model1_img.cols, model1_img.rows)));
		model1_img = white_mat.clone();
	}
	//然后开始正常的计算，求定位以及坐标点

	int rows, cols;
	std::vector<cv::Rect> rects = iniRects(512, 512, model1_img.rows, model1_img.cols, 128, true, true, rows, cols);
	std::vector<cv::Mat> imgs;
	for (auto elem : rects)
	{
		imgs.emplace_back(model1_img(elem).clone());
	}
	tr_model_holder->tr_model1->processDataConcurrency(imgs);
	auto results = tr_model_holder->tr_model1->m_results;

	//model2需要在raw_img上取图像，因此，需要将model1的定位点还原到原始坐标上
	std::vector<cv::Mat> m2_imgs;
	std::vector<cv::Point> m2_points;
	for (auto iter = results.begin(); iter != results.end(); iter++)
	{
		int place = iter - results.begin();
		if (iter->score > 0.5f)
		{
			for (int i = 1; i < iter->points.size(); i++)
			{
				auto elem = iter->points[i];
				//以当前点为中心取一定大小的图像
				cv::Mat temp_img;
				int center_x = (rects[place].x + elem.x) * float(model1_mpp / img_mpp);
				int center_y = (rects[place].y + elem.y) * float(model1_mpp / img_mpp);
				int get_height = model2_height * float(model2_mpp / img_mpp);
				int get_width = model2_width * float(model2_mpp / img_mpp);

				getSubMat(raw_img, temp_img, center_x, center_y, get_height, get_width);
				cv::resize(temp_img, temp_img, cv::Size(model2_width, model2_height));
				m2_points.emplace_back(cv::Point(center_x, center_y));
				m2_imgs.emplace_back(std::move(temp_img));
			}
		}
	}
	if (m2_imgs.size() == 0)
		return;

	tr_model_holder->tr_model2->processDataConcurrency(m2_imgs);
	auto results2 = tr_model_holder->tr_model2->m_results;

	std::vector<PointScore> model2_ps;
	auto lambda = [](PointScore ps1, PointScore ps2)->bool {
		if (ps1.score > ps2.score) {
			return true;
		}
		return false;
	};
	for (int i = 0; i < results2.size(); i++)
	{
		PointScore temp_ps;
		temp_ps.point = m2_points[i];
		temp_ps.score = results2[i].score;
		model2_ps.emplace_back(temp_ps);
	}
	std::sort(model2_ps.begin(), model2_ps.end(), lambda);
	if (model2_ps.size() > n)
	{
		model2_ps.erase(model2_ps.begin() + n, model2_ps.end());
	}
	//然后去重(根据50um进行去重)
	filterBaseOnPoint(model2_ps, 25.0f / img_mpp);
	//将model2_ps放到anno里面
	int i = 0;
	for (auto elem : model2_ps)
	{
		Anno anno;
		anno.id = i;
		anno.x = elem.point.x;
		anno.y = elem.point.y;
		anno.score = elem.score;
		annos.emplace_back(anno);
		i++;
	}
}

void freeModelMem(ArHandle myHandle)
{
	TrModelHolder* tr_model_holder = (TrModelHolder*)myHandle;
	delete tr_model_holder->tr_model1;
	tr_model_holder->tr_model1 = nullptr;
	delete tr_model_holder->tr_model2;
	tr_model_holder->tr_model2 = nullptr;
	delete tr_model_holder;
	tr_model_holder = nullptr;
}

//
void writeAnnos2Img(std::vector<Anno> annos, 
	cv::Mat& img, 
	std::string img_save_path, 
	std::string save_name, 
	double mpp)
{
	if (annos.size() == 0)
		return;
	for (auto elem : annos)
	{
		cv::circle(img, cv::Point(elem.x, elem.y), 25 / mpp, cv::Scalar(0, 0, 255), 4);
		cv::putText(img, std::to_string(elem.score), cv::Point(elem.x, elem.y), 1, 1, cv::Scalar(0, 255, 0), 2);
	}
	cv::imwrite(img_save_path + save_name, img);
}

void agetARConfigByPara(ARConfig& ar_config, char** argv)
{
	ar_config.img_mpp = std::atof(argv[2]);
	ar_config.img_path = std::string(argv[3]);
	ar_config.img_save_path = std::string(argv[4]);
	ar_config.img_suffix = std::string(argv[5]);
	ar_config.max_recom_num = std::atoi(argv[6]);
	ar_config.cuda_visible_devices = std::string(argv[7]);
}

void getARConfig(ARConfig& ar_config, std::string config_path)
{
	setIniPath(config_path);

	ar_config.img_mpp = IniConfig::instance().getIniDouble("AR", "img_mpp");
	ar_config.img_path = IniConfig::instance().getIniString("AR", "img_path");
	ar_config.img_save_path = IniConfig::instance().getIniString("AR", "img_save_path");
	ar_config.img_suffix = IniConfig::instance().getIniString("AR", "img_suffix");
	ar_config.max_recom_num = IniConfig::instance().getIniInt("AR", "max_recom_num");
	ar_config.cuda_visible_devices = IniConfig::instance().getIniString("AR", "CUDA_VISIBLE_DEVICES");
}

//发现不仅仅有ar的配置，还包含了TRT的配置，因此还是要传入配置文件的路径，否则你想一个个的传入TRT的路径和配置吗？
void startRun(ARConfig ar_config, std::string config_path)
{
	_putenv_s("CUDA_VISIBLE_DEVICES", ar_config.cuda_visible_devices.c_str());
	std::vector<std::string> cut_imgs;
	getFiles(ar_config.img_path, cut_imgs, ar_config.img_suffix);

	if (!std::filesystem::exists(ar_config.img_save_path))
	{
		std::filesystem::create_directories(ar_config.img_save_path);
	}

	ArHandle handle = initialize_handle(config_path.c_str());

	int i = 0;
	int total = cut_imgs.size();
	for (auto cut_img : cut_imgs)
	{
		std::cout << i << ":" << total << " ";
		std::vector<Anno> annos;
		cv::Mat img = cv::imread(cut_img);
		std::string img_name = getFileName(cut_img);
		process(annos, handle, img, ar_config.img_mpp, ar_config.max_recom_num);
		writeAnnos2Img(annos, img, ar_config.img_save_path, img_name, ar_config.img_mpp);
		i++;
	}
	freeModelMem(handle);
}

int main(int args, char** argv)
{
	ARConfig ar_config;
	std::string config_path = "";
	switch (args)
	{
	case 1:
		//表示本地测试
		config_path = "./config.ini";
		getARConfig(ar_config, config_path);
		break;
	case 2:
		//表示只使用config.ini来完成
		config_path = std::string(argv[1]);
		getARConfig(ar_config, argv[1]);
		break;
	case 7:
		//表示AR部分的要使用传入的参数来运行
		config_path = std::string(argv[1]);
		agetARConfigByPara(ar_config, argv);
		break;
	default:
		std::cout << "please check your parameter\n";
	}
	ar_config.showARConfig();
	startRun(ar_config, config_path);
	system("pause");
	return 0;
}