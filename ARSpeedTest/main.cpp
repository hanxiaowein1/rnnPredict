#include "TrModel1.h"
#include "TrModel2.h"
#include "IniConfig.h"
#include "commonFunction.h"
#include <filesystem>
#include <chrono>
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
vector<cv::Rect> iniRects(
	int sHeight, int sWidth, int height, int width,
	int overlap, bool flag_right, bool flag_down,
	int& rows, int& cols)
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

//取子图，如果越界，则填黑
void getSubMat(cv::Mat& src, cv::Mat& dst, int center_x, int center_y, int height, int width)
{
	//判断center_x和center_y是否越界
	//if(center_x > src.rows || center_x < 0 || center_ > )

	//没想到被奇偶搞翻车了，如果是奇数的话，除以二之后会少一个坐标点，因此，如果为奇数的话，就在下面和右边多裁一个坐标点

	int half_height = height / 2;
	int half_width = width / 2;
	int height_is_odd = height % 2;
	int width_is_odd = width % 2;
	
	int rows = src.rows;
	int cols = src.cols;
	int top = 0, bottom = 0, left = 0, right = 0;
	top = (center_y - half_height) > 0 ? (center_y - half_height) : 0;
	bottom = (center_y + half_height - 1 + height_is_odd) >= rows ? (rows - 1) : (center_y + half_height - 1 + height_is_odd);
	left = (center_x - half_width) > 0 ? (center_x - half_width) : 0;
	right = (center_x + half_width - 1 + width_is_odd) >= cols ? (cols - 1) : (center_x + half_width - 1 + width_is_odd);

	cv::Rect rect(left, top, right - left + 1, bottom - top + 1);
	cv::Mat tmp = src(rect);

	int topStick = (center_y - half_height) > 0 ? 0 : abs(center_y - half_height);
	int leftStick = (center_x - half_width) > 0 ? 0 : abs(center_x - half_width);
	cv::Mat white_mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
	tmp.copyTo(white_mat(cv::Rect(leftStick, topStick, tmp.cols, tmp.rows)));

	dst = white_mat.clone();
}

void filterBaseOnPoint(vector<PointScore>& PointScores, int threshold)
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

//更改需求，适应进入的图像mpp，得到定位点和分数
void computeArbitraryImg(
	std::string img_path,
	TrModel1& tr_model1,
	TrModel2& tr_model2,
	std::string save_path,
	double img_mpp,
	int n)
{
	double model1_mpp    = 0.586f;
	int    model1_width  = 512;
	int    model1_height = 512;
	double model2_mpp    = 0.293f;
	int    model2_width  = 256;
	int    model2_height = 256;

	cv::Mat raw_img = cv::imread(img_path);

	auto start = std::chrono::system_clock::now();


	//先获取model1的图像
	if (!std::filesystem::exists(save_path))
		std::filesystem::create_directories(save_path);
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
		int new_width  = model1_img.cols < 512 ? 512 : model1_img.cols;
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
	tr_model1.processDataConcurrency(imgs);
	auto results = tr_model1.m_results;
	
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
				//以当前点为中心取128*128的图像
				cv::Mat temp_img;
				//cv::Rect rect(elem.x, elem.y, );
				int center_x = (rects[place].x + elem.x) * float(model1_mpp / img_mpp);
				int center_y = (rects[place].y + elem.y) * float(model1_mpp / img_mpp);
				int get_height = model2_height * float(model2_mpp / img_mpp);
				int get_width  = model2_width  * float(model2_mpp / img_mpp);

				getSubMat(raw_img, temp_img, center_x, center_y, get_height, get_width);
				cv::resize(temp_img, temp_img, cv::Size(model2_width, model2_height));
				m2_points.emplace_back(cv::Point(center_x, center_y));
				m2_imgs.emplace_back(std::move(temp_img));
			}
		}
	}
	if (m2_imgs.size() == 0)
		return;

	tr_model2.processDataConcurrency(m2_imgs);
	auto results2 = tr_model2.m_results;

	std::vector<PointScore> model2_ps;
	auto lambda = [](PointScore ps1, PointScore ps2)->bool {
		if (ps1.score > ps2.score) {
			return true;
		}
		return false;
	};
	for (int i = 0; i < results.size(); i++)
	{
		PointScore temp_ps;
		temp_ps.point = m2_points[i];
		temp_ps.score = results[i].score;
		model2_ps.emplace_back(temp_ps);
	}
	std::sort(model2_ps.begin(), model2_ps.end(), lambda);
	if (model2_ps.size() > n)
	{
		model2_ps.erase(model2_ps.begin() + n, model2_ps.end());
	}
	//然后去重(根据50um进行去重)
	filterBaseOnPoint(model2_ps, 25.0f / img_mpp);



	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	cout
		<< double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den
		<< endl;
	//把点和分数画到opencv上
	for (int i = 0; i < model2_ps.size(); i++)
	{
		cv::circle(raw_img, model2_ps[i].point, 25 / img_mpp, cv::Scalar(0, 0, 255), 4);
		cv::putText(raw_img, to_string(model2_ps[i].score), model2_ps[i].point, 1, 1, cv::Scalar(0, 255, 0), 2);
	}
	std::filesystem::path fs_path(img_path);
	//return fs_path.filename().string();
	std::string save_img_name = fs_path.filename().string();
	cv::imwrite(save_path + "/" + save_img_name, raw_img);
}

//计算单张图像
void computeSingleImg(std::string img_path, TrModel1& tr_model1, TrModel2& tr_model2, std::string save_path)
{
	if (!std::filesystem::exists(save_path))
		std::filesystem::create_directories(save_path);
	cv::Mat img = cv::imread(img_path);

	auto start = std::chrono::system_clock::now();

	int cols = 0;
	int rows = 0;
	std::vector<cv::Rect> rects = iniRects(512, 512, img.rows, img.cols, 128, true, true, rows, cols);
	std::vector<cv::Mat> imgs;
	for (auto elem : rects)
	{
		imgs.emplace_back(img(elem).clone());
	}
	tr_model1.processDataConcurrency(imgs);
	auto results = tr_model1.m_results;

	//std::vector<std::pair<cv::Point, cv::Mat>> m2_imgs;
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
				//以当前点为中心取128*128的图像
				cv::Mat temp_img;
				int center_x = rects[place].x + elem.x;
				int center_y = rects[place].y + elem.y;
				getSubMat(img, temp_img, center_x, center_y, 128, 128);
				cv::resize(temp_img, temp_img, cv::Size(256, 256));
				m2_points.emplace_back(cv::Point(center_x, center_y));
				m2_imgs.emplace_back(std::move(temp_img));
			}
		}
	}
	if (m2_imgs.size() == 0)
		return;
	tr_model2.processDataConcurrency(m2_imgs);
	auto results2 = tr_model2.m_results;

	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	cout 
		<< double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den 
		<< endl;
	//把点和分数画到opencv上
	for (int i = 0; i < results2.size(); i++)
	{
		cv::circle(img, m2_points[i], 50, cv::Scalar(0, 0, 255), 4);
		cv::putText(img, to_string(results2[i].score), m2_points[i], 1, 1, cv::Scalar(0, 255, 0), 2);
	}
	std::filesystem::path fs_path(img_path);
	//return fs_path.filename().string();
	std::string save_img_name = fs_path.filename().string();
	cv::imwrite(save_path + "/" + save_img_name, img);
}


//int main()
//{
//	//setIniPath("../x64/Release/config.ini");
//	
//	_putenv_s("CUDA_VISIBLE_DEVICES", "0");
//	setIniPath("./config.ini");
//	TrModel1 tr_model1("TrModel1");
//	tr_model1.createThreadPool(1);
//	TrModel2 tr_model2("TrModel2");
//	tr_model2.createThreadPool(1);
//	std::string slide_path = "D:\\TEST_OUTPUT\\裁图\\";
//	std::string save_path = "D:\\TEST_OUTPUT\\rnnPredict\\ArSpeed\\";
//	std::vector<std::string> slides = getDirs(slide_path);
//	for (auto elem : slides)
//	{
//		std::string file_name = getFileName(elem);
//		//从文件夹中得到mpp
//		std::vector<std::string> split_str = split(file_name, '_');
//		double img_mpp = std::stod(split_str[split_str.size() - 1]);
//		std::vector<std::string> cut_imgs;
//		getFiles(elem, cut_imgs, "tif");
//		for (auto cut_img : cut_imgs)
//		{
//			computeArbitraryImg(cut_img, tr_model1, tr_model2, save_path + file_name, img_mpp, 10);
//			//computeSingleImg(cut_img, tr_model1, tr_model2, save_path + file_name);
//		}
//	}
//	system("pause");
//	return 0;
//}