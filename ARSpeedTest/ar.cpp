#include "ar.h"

#include "commonFunction.h"
#include "model_holder.h"
#include <iostream>
#include <filesystem>
#include <map>

extern vector<cv::Rect> iniRects(
	int sHeight, int sWidth, int height, int width,
	int overlap, bool flag_right, bool flag_down,
	int& rows, int& cols);

extern void getSubMat(cv::Mat& src, cv::Mat& dst, int center_x, int center_y, int height, int width);
extern void filterBaseOnPoint(vector<PointScore>& PointScores, int threshold);

ArHandle initialize_handle(std::string ini_path)
{
	setIniPath(ini_path.c_str());
	if (IniConfig::instance().getIniString("TensorRT", "USE_TR") == "OFF") {
		//使用tensorflow
		TfModelHolder* tf_model_holder = new TfModelHolder;
		tf_model_holder->tf_model1 = new TfModel1("TfModel1");
		tf_model_holder->tf_model1->createThreadPool(1);
		tf_model_holder->tf_model2 = new TfModel2("TfModel2");
		tf_model_holder->tf_model2->createThreadPool(1);
		return ArHandle(tf_model_holder);
	}
	TrModelHolder* tr_model_holder = new TrModelHolder;
	tr_model_holder->tr_model1 = new TrModel1("TrModel1");
	tr_model_holder->tr_model1->createThreadPool(1);
	tr_model_holder->tr_model2 = new TrModel2("TrModel2");
	tr_model_holder->tr_model2->createThreadPool(1);
	return ArHandle(tr_model_holder);
}

void process(
	std::vector<Anno>& annos, 
	ArHandle myHandle, 
	cv::Mat& raw_img, 
	double img_mpp, int n, double score_threshold, double remove_threshold)
{
	auto model_holder = (ModelHolder*)myHandle;
	double model1_mpp = IniConfig::instance().getIniDouble("Model1", "mpp");
	int    model1_width = IniConfig::instance().getIniInt("Model1", "width");
	int    model1_height = IniConfig::instance().getIniInt("Model1", "height");
	double model2_mpp = IniConfig::instance().getIniDouble("Model2", "mpp");
	int    model2_width = IniConfig::instance().getIniInt("Model2", "width");
	int    model2_height = IniConfig::instance().getIniInt("Model2", "height");

	auto start = std::chrono::system_clock::now();


	//先获取model1的图像
	cv::Mat model1_img;
	if (img_mpp != model1_mpp)
	{
		int new_height = raw_img.rows / float(model1_mpp / img_mpp);
		int new_width = raw_img.cols / float(model1_mpp / img_mpp);
		cv::resize(raw_img, model1_img, cv::Size(new_width, new_height));
	}
	else
		model1_img = raw_img;
	//如果model1_img的宽高并没有达到512，那么我就补，哪边不够补哪边
	if (model1_img.rows < model1_height || model1_img.cols < model1_width)
	{
		//开始补
		int new_height = model1_img.rows < model1_height ? model1_height : model1_img.rows;
		int new_width = model1_img.cols < model1_width ? model1_width : model1_img.cols;
		cv::Mat white_mat(new_height, new_width, CV_8UC3, cv::Scalar(0, 0, 0));
		model1_img.copyTo(white_mat(cv::Rect(0, 0, model1_img.cols, model1_img.rows)));
		model1_img = white_mat.clone();
	}
	//然后开始正常的计算，求定位以及坐标点

	int rows, cols;
	std::vector<cv::Rect> rects = iniRects(
		model1_height, model1_width, model1_img.rows, model1_img.cols, model1_height / 4, true, true, rows, cols);
	std::vector<cv::Mat> imgs;
	for (auto elem : rects)
	{
		imgs.emplace_back(model1_img(elem).clone());
	}
	model_holder->processDataConcurrencyM1(imgs);
	auto results = model_holder->getModel1Result();
	//tr_model_holder->tr_model1->processDataConcurrency(imgs);
	//auto results = tr_model_holder->tr_model1->m_results;

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
				m2_points.emplace_back(cv::Point(center_x, center_y)																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																	);
				m2_imgs.emplace_back(std::move(temp_img));
			}
		}
	}
	if (m2_imgs.size() == 0)
		return;

	model_holder->processDataConcurrencyM2(m2_imgs);
	auto results2 = model_holder->getModel2Result();
	//tr_model_holder->tr_model2->processDataConcurrency(m2_imgs);
	//auto results2 = tr_model_holder->tr_model2->m_results;

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
	filterBaseOnPoint(model2_ps, remove_threshold / img_mpp);
	//将model2_ps放到anno里面
	int i = 0;
	for (auto elem : model2_ps)
	{
		if (elem.score < score_threshold)
			continue;
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
	cv::imwrite(img_save_path + "\\" + save_name, img);
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
	ar_config.remove_threshold = IniConfig::instance().getIniDouble("AR", "remove_threshold");
	ar_config.score_threshold = IniConfig::instance().getIniDouble("AR", "score_threshold");
}

//发现不仅仅有ar的配置，还包含了TRT的配置，因此还是要传入配置文件的路径，否则你想一个个的传入TRT的路径和配置吗？
void startRun(ARConfig ar_config, std::string config_path)
{
	using namespace std;
	_putenv_s("CUDA_VISIBLE_DEVICES", ar_config.cuda_visible_devices.c_str());
	ArHandle handle = (ArHandle) new ModelHolder(config_path);
	RnnHolder* rnnHolder = new RnnHolder(config_path);
	std::string slidePath = ar_config.img_path;
	std::string savePath = ar_config.img_save_path;
	if (std::filesystem::exists(savePath)) {
		std::filesystem::create_directories(savePath);
	}
	//编写一个能够计算所有文件夹的程序
	vector<string> img_dirs;
	img_dirs.emplace_back(slidePath);
	for (auto& p : std::filesystem::recursive_directory_iterator(ar_config.img_path)) 
	{
		if (std::filesystem::is_directory(p)) {
			img_dirs.emplace_back(p.path().string());
		}
	}

	vector<string> img_suffixs{ "tif", "png", "PNG", "jpg", "jpeg", "bmp" };

	for (auto dir : img_dirs)
	{
		std::cout << "running " << dir << std::endl;
		auto position = dir.find(slidePath);
		string dir_cp = dir;
		dir_cp.erase(position, ar_config.img_path.size());
		string temp_save_path = savePath + dir_cp;
		if (!std::filesystem::exists(temp_save_path)) {
			std::filesystem::create_directories(temp_save_path);
		}
		else 
		{
			//检查是否有去重文件，如果有则跳过该张切片
			if (filesystem::exists(temp_save_path + "\\" + "rnn.txt"))
			{
				continue;
			}
		}
		std::vector<std::string> cut_imgs;
		for (auto img_suffix : img_suffixs)
		{
			std::vector<std::string> temp_imgs = getFilesBySuffix(dir, img_suffix);
			cut_imgs.insert(cut_imgs.end(), temp_imgs.begin(), temp_imgs.end());
		}
		//getFiles(dir, cut_imgs, ar_config.img_suffix);

		vector<pair<string, Anno>> all_img_anno;

		int i = 0;
		int total = cut_imgs.size();
		for (auto cut_img : cut_imgs)
		{
			//pair<string, Anno> img_annos;
			std::cout << i << ":" << total << " ";
			std::vector<Anno> annos;
			cv::Mat img = cv::imread(cut_img);
			std::string img_name = getFileName(cut_img);
			process(
				annos,
				handle,
				img,
				ar_config.img_mpp,
				ar_config.max_recom_num,
				ar_config.score_threshold,
				ar_config.remove_threshold);

			for (auto anno : annos)
			{
				pair<string, Anno> img_anno;
				img_anno.first = cut_img;
				img_anno.second = anno;
				all_img_anno.emplace_back(img_anno);
			}
			//writeAnnos2Img(annos, img, ar_config.img_save_path, img_name, ar_config.img_mpp);
			i++;
		}
		//再将这个all_img_annos进行排序(从大到小进行排序)
		auto lambda = [](pair<string, Anno> a, pair<string, Anno> b) -> bool {
			if (a.second.score > b.second.score) {
				return true;
			}
			return false;
		};
		std::sort(all_img_anno.begin(), all_img_anno.end(), lambda);
		ofstream rnn_txt;
		rnn_txt.open(temp_save_path + "\\" + "rnn.txt");
		if (all_img_anno.size() < 30)
		{
			std::cout << "not enough annos for compute rnn!" << std::endl;
			rnn_txt << "0";
			rnn_txt.close();
			continue;
		}
		else
		{
			all_img_anno.erase(all_img_anno.begin() + 30, all_img_anno.end());
		}
		map<string, std::vector<Anno>> recom_img_annos;
		//将前十个annos的图像送入rnn中进行计算得到结果
		vector<cv::Mat> model2_imgs;
		int count = 0;

		for (auto elem : all_img_anno)
		{
			auto img_path = elem.first;
			auto anno = elem.second;
			auto raw_img = cv::imread(img_path);

			double model2_mpp    = IniConfig::instance().getIniDouble("Model2", "mpp");;
			int    model2_width  = IniConfig::instance().getIniInt("Model2", "width");
			int    model2_height = IniConfig::instance().getIniInt("Model2", "height");
			//为了和lsb的逻辑一致，这里先将图像resize到model2的分辨率下
			int dst_height = raw_img.rows * (ar_config.img_mpp / model2_mpp);
			int dst_width = raw_img.cols * (ar_config.img_mpp / model2_mpp);
			//cv::resize(raw_img, raw_img, cv::Size(dst_width, dst_height));

			double ratio = ar_config.img_mpp / model2_mpp;
			cv::resize(raw_img, raw_img, cv::Size(0, 0), ratio, ratio);

			int center_x = anno.x * (ar_config.img_mpp / model2_mpp);
			int center_y = anno.y * (ar_config.img_mpp / model2_mpp);
			cv::Mat temp_img;
			getSubMat(raw_img, temp_img, center_x, center_y, model2_height, model2_width);


			//int get_height = model2_height * float(model2_mpp / ar_config.img_mpp);
			//int get_width = model2_width * float(model2_mpp / ar_config.img_mpp);
			//cv::Mat temp_img;
			//getSubMat(raw_img, temp_img, anno.x, anno.y, get_height, get_width);
			//cv::resize(temp_img, temp_img, cv::Size(model2_width, model2_height));

			//把这些图像保存到文件夹中
			std::string model2InputSavePath = temp_save_path + "\\model2";
			if (!std::filesystem::exists(model2InputSavePath)) {
				std::filesystem::create_directories(model2InputSavePath);
			}
			cv::imwrite(model2InputSavePath + "\\" + to_string(count) + "_" + to_string(anno.score) + ".tif", temp_img);
			model2_imgs.emplace_back(std::move(temp_img));

			if (count < 10)
			{
				recom_img_annos[img_path].emplace_back(elem.second);
			}
			count++;
		}
		for (auto elem : recom_img_annos)
		{
			cv::Mat img = cv::imread(elem.first);
			auto img_name = getFileName(elem.first);
			writeAnnos2Img(elem.second, img, temp_save_path, img_name, ar_config.img_mpp);
			count++;
		}
		//为了防止重复计算，在这里去重


		auto modelHolder = (ModelHolder*)handle;

		modelHolder->processDataConcurrencyM2(model2_imgs);
		auto results = modelHolder->getModel2Result();

		
		float rnnScore = rnnHolder->runRnn(results);
		std::cout << "rnnScore is" << rnnScore << std::endl;
		rnn_txt << rnnScore;
		rnn_txt.close();

	}

	//ArHandle handle = initialize_handle(config_path.c_str());

	//保存着img的annos的vector
	//vector<pair<string, vector<Anno>>> all_img_annos;
	auto modelHolder = (ModelHolder*)handle;
	delete rnnHolder;
	delete modelHolder;
	//freeModelMem(handle);
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