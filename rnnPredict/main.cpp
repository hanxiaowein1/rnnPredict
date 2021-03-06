#include <iostream>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <windows.h>
#include "commonFunction.h"
#include "SlideProc.h"
#include "DLLManager.h"
#include "IniConfig.h"
#include "DetectDll.h"
void concatTest()
{
	using namespace tensorflow::ops;
	tensorflow::Scope root = tensorflow::Scope::NewRootScope();
	auto t1 = Const(root, { {1.f, 2.f}, {3.f, 4.f} });
	auto t2 = Const(root, { {5.f, 6.f}, {7.f, 8.f}, {9.f, 0.f} });
	auto concatT1T2 = Concat(root.WithOpName("ConcatT1T2"), { t1, t2 }, 0);
	std::vector<tensorflow::Tensor> outputs;
	tensorflow::ClientSession session(root);
	TF_CHECK_OK(session.Run({ concatT1T2 }, &outputs));
	// Get output tensor
	tensorflow::Tensor result = outputs[0];
	// Print output
	LOG(INFO) << result.matrix<float>();
}

void GammaCorrection(cv::Mat& src, cv::Mat& dst, float fGamma)
{
	unsigned char lut[256];
	for (int i = 0; i < 256; i++) {
		lut[i] = cv::saturate_cast<uchar>(int(pow((float)(i / 255.0), fGamma) * 255.0f));
	}
	dst = src.clone();
	const int channels = dst.channels();
	switch (channels) {
	case 1: {
		cv::MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
			*it = lut[(*it)];
		break;
	}
	case 3: {
		for (int i = 0; i < dst.rows; i++) {
			uchar* linePtr = dst.ptr(i);
			for (int j = 0; j < dst.cols; j++) {
				*(linePtr + j * 3) = lut[*(linePtr + j * 3)];
				*(linePtr + j * 3 + 1) = lut[*(linePtr + j * 3 + 1)];
				*(linePtr + j * 3 + 2) = lut[*(linePtr + j * 3 + 2)];
			}
		}
		break;
	}
	}
}

vector<cv::Rect> getRects(int srcImgWidth, int srcImgHeight, int dstImgWidth, int dstImgHeight, int m, int n)
{
	vector<cv::Rect> myRects;
	//计算每次裁剪的间隔(hDirect,wDirect)
	int wDirect = (srcImgWidth - dstImgWidth) / (m - 1);
	int hDirect = (srcImgHeight - dstImgHeight) / (n - 1);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			int topValue = i * hDirect;
			int leftValue = j * wDirect;
			cv::Rect myRect(leftValue, topValue, dstImgWidth, dstImgHeight);
			myRects.push_back(myRect);
		}
	}
	return myRects;
}

//如果最后有剩余，不裁取和前面相同的大小，而是仍然保持相同冗余，最后有多少裁多少
vector<cv::Rect> iniRects(int sHeight, int sWidth, int height, int width, int overlap)
{
	vector<cv::Rect> rects;
	//进行参数检查
	if (sHeight == 0 || sWidth == 0 || height == 0 || width == 0)
	{
		cout << "iniRects: parameter should not be zero\n";
		return rects;
	}
	if (sHeight > height || sWidth > width)
	{
		cout << "iniRects: sHeight or sWidth > height or width\n";
		return rects;
	}
	int x_num = (width - overlap) / (sWidth - overlap);
	int y_num = (height - overlap) / (sHeight - overlap);
	vector<int> xStart;
	vector<int> yStart;
	bool flag_right = true;
	bool flag_down = true;
	if ((x_num * (sWidth - overlap) + overlap) == width)
	{
		flag_right = false;
	}
	if ((y_num * (sHeight - overlap) + overlap) == height)
	{
		flag_down = false;
	}
	for (int i = 0; i < x_num; i++)
	{
		xStart.emplace_back((sWidth - overlap) * i);
	}
	for (int i = 0; i < y_num; i++)
	{
		yStart.emplace_back((sHeight - overlap) * i);
	}
	if (flag_right)
		xStart.emplace_back((sWidth - overlap) * x_num);
	if (flag_down)
		yStart.emplace_back((sHeight - overlap) * y_num);
	int last_width = width - x_num * (sWidth - overlap);
	int last_height = height - y_num * (sHeight - overlap);
	for (int i = 0; i < yStart.size(); i++)
	{
		for (int j = 0; j < xStart.size(); j++)
		{
			cv::Rect rect;
			rect.x = xStart[j];
			rect.y = yStart[i];
			rect.width = sWidth;
			rect.height = sHeight;
			if (i == yStart.size() - 1)
			{
				rect.height = last_height;
			}
			if (j == xStart.size() - 1)
			{
				rect.width = last_width;
			}
			rects.emplace_back(rect);
		}
	}
	return rects;
}

//如果flag_right为true，那么就靠右多裁一个；
//如果flag_down为true，那么就靠下多裁一个。
vector<cv::Rect> iniRects(int sHeight, int sWidth, int height, int width, int overlap, bool flag_right, bool flag_down)
{
	vector<cv::Rect> rects;
	//进行参数检查
	if (sHeight == 0 || sWidth == 0 || height == 0 || width == 0)
	{
		cout << "iniRects: parameter should not be zero\n";
		return rects;
	}
	if (sHeight > height || sWidth > width)
	{
		cout << "iniRects: sHeight or sWidth > height or width\n";
		return rects;
	}
	int x_num = (width - overlap) / (sWidth - overlap);
	int y_num = (height - overlap) / (sHeight - overlap);
	vector<int> xStart;
	vector<int> yStart;
	if ((x_num * (sWidth - overlap) + overlap) == width)
	{
		flag_right = false;
	}
	if ((y_num * (sHeight - overlap) + overlap) == height)
	{
		flag_down = false;
	}
	for (int i = 0; i < x_num; i++)
	{
		xStart.emplace_back((sWidth - overlap) * i);
	}
	for (int i = 0; i < y_num; i++)
	{
		yStart.emplace_back((sHeight - overlap) * i);
	}
	if(flag_right)
		xStart.emplace_back(width - sWidth);
	if(flag_down)
		yStart.emplace_back(height - sHeight);
	for (int i = 0; i < yStart.size(); i++)
	{
		for (int j = 0; j < xStart.size(); j++)
		{
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

vector<cv::Rect> iniRects(int sHeight, int sWidth, int height, int width)
{
	vector<cv::Rect> rects;
	//进行参数检查
	if (sHeight == 0 || sWidth == 0 || height == 0 || width == 0)
	{
		cout << "iniRects: parameter should not be zero\n";
		return rects;
	}
	if (sHeight > height || sWidth > width)
	{
		cout << "iniRects: sHeight or sWidth > height or width\n";
		return rects;
	}
	int xNum = 0;//水平方向的裁剪个数
	int yNum = 0;//垂直方向的裁剪个数
	int overlap = 1272 * (0.25f);
	if (sHeight <= overlap || sWidth <= overlap)
	{
		cout << "sHeight and sWidth seems to small\n";
		return rects;
	}
	yNum = 1 + (height - sHeight) / (sHeight - overlap);
	xNum = 1 + (width - sWidth) / (sWidth - overlap);
	vector<int> xStart;
	vector<int> yStart;
	for (int i = 0; i < xNum; i++)
	{
		xStart.emplace_back((sWidth - overlap) * i);
	}
	for (int i = 0; i < yNum; i++)
	{
		yStart.emplace_back((sHeight - overlap) * i);
	}
	int xLeft = width - xNum * sWidth;
	int yLeft = height - yNum * sHeight;
	if (xLeft != 0)
		xStart.emplace_back(width - sWidth - 1);
	if (yLeft != 0)
		yStart.emplace_back(height - sHeight - 1);
	for (int i = 0; i < yStart.size(); i++)
	{
		for (int j = 0; j < xStart.size(); j++)
		{
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

vector<cv::Rect> iniRects(MultiImageRead& mImgRead)
{

	double mpp = 0.0f;
	mImgRead.getSlideMpp(mpp);
	int height = 0;
	mImgRead.getSlideHeight(height);
	int width = 0;
	mImgRead.getSlideWidth(width);
	int sHeight = 1936 * float(0.586f / mpp);
	int sWidth = 1216 * float(0.586f / mpp);
	vector<cv::Rect> rects;
	if (height == 0 || width == 0 || sHeight == 0 || sWidth == 0)
	{
		cout << "iniRects: some parameters should not be zero\n";
		return rects;
	}
	if (height < sHeight || width < sWidth)
	{
		cout << "size to be cropped should bigger \n";
		return rects;
	}
	int xNum = 0;//水平方向的裁剪个数
	int yNum = 0;//垂直方向的裁剪个数
	int overlap = 120 * float(0.586f / mpp);
	if (sHeight <= overlap || sWidth <= overlap)
	{
		cout << "sHeight and sWidth seems to small\n";
		return rects;
	}
	yNum = 1 + (height - sHeight) / (sHeight - overlap);
	xNum = 1 + (width - sWidth) / (sWidth - overlap);
	vector<int> xStart;
	vector<int> yStart;
	for (int i = 0; i < xNum; i++)
	{
		xStart.emplace_back((sWidth - overlap) * i);
	}
	for (int i = 0; i < yNum; i++)
	{
		yStart.emplace_back((sHeight - overlap) * i);
	}
	int xLeft = width - xNum * sWidth;
	int yLeft = height - yNum * sHeight;
	if (xLeft != 0)
		xStart.emplace_back(width - sWidth - 1);
	if (yLeft != 0)
		yStart.emplace_back(height - sHeight - 1);
	for (int i = 0; i < yStart.size(); i++)
	{
		for (int j = 0; j < xStart.size(); j++)
		{
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

void output2(std::vector<cv::Mat>& imgs)
{
	int batchsize = imgs.size();
	tensorflow::Tensor tem_tensor_res(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({ batchsize, imgs[0].rows, imgs[0].cols, 3 }));
	auto mapTensor = tem_tensor_res.tensor<float, 4>();

	for (int i = 0; i < imgs.size(); i++)
	{
		float* ptr = tem_tensor_res.flat<float>().data() + i * imgs[0].rows * imgs[0].cols * 3;
		cv::Mat tensor_image(imgs[0].rows, imgs[0].cols, CV_32FC3, ptr);
		imgs[i].convertTo(tensor_image, CV_32F);//转为float类型的数组
		tensor_image = (tensor_image / 255 - 0.5) * 2;
	}
}

void output(std::vector<cv::Mat>& imgs)
{
	int batchsize = imgs.size();
	tensorflow::Tensor tem_tensor_res(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({ batchsize, imgs[0].rows, imgs[0].cols, 3 }));
	auto mapTensor = tem_tensor_res.tensor<float, 4>();

	for (int batch = 0; batch < batchsize; batch++)
	{
		for (int h = 0; h < imgs[0].rows; h++)
		{
			auto linePtr = imgs[batch].ptr(h);
			for (int w = 0; w < imgs[0].cols; w++)
			{
				mapTensor(batch, h, w, 0) = (((*(linePtr + w * 3))*1.0f) / 255 - 0.5) * 2;
				mapTensor(batch, h, w, 1) = (((*(linePtr + w * 3 + 1))*1.0f) / 255 - 0.5) * 2;
				mapTensor(batch, h, w, 2) = (((*(linePtr + w * 3 + 2))*1.0f) / 255 - 0.5) * 2;
			}
		}
	}
}

//这个是给刘思博的，测试所有切片的，不过model2要改成0.243，model1要改成0.486
//void subTask2(vector<string>& slideList, SlideProc& slideProc, string savePath)
//{
//	for (auto& elem : slideList)
//	{
//		cout << elem << " start processing\n";
//		string filename = getFileNamePrefix(&elem);
//		//在这里为每一个片子单独生成一个文件夹
//		if (!slideProc.runSlide2(elem.c_str()))
//			continue;
//		slideProc.saveResult(savePath, filename);
//		slideProc.saveResult2(savePath, filename);
//	}
//}

//这个是给学姐的model3
void subTask(vector<string> &slideList, SlideProc &slideProc, string savePath)
{
	for (auto& elem : slideList)
	{
		cout << elem << " start processing\n";
		string filename = getFileNamePrefix(&elem);
		//在这里为每一个片子单独生成一个文件夹
		string finalSavePath = string(savePath) + "\\" + filename;
		createDirRecursive(finalSavePath);
		if (!slideProc.runSlide3(elem.c_str(), finalSavePath))
			continue;
		slideProc.saveResult(savePath, filename);
		slideProc.saveResult2(savePath, filename);
		slideProc.saveResult3(savePath, filename + "_rnnScore");
	}
}

void startRun(const char* iniPath)
{
	DLLManager manager;
	SlideProc slideProc(iniPath);
	std::string slidePath = IniConfig::instance().getIniString("Path", "slidePath");
	std::string savePath = IniConfig::instance().getIniString("Path", "savePath");
	if (std::filesystem::exists(savePath)) {
		std::filesystem::create_directories(savePath);
	}
	//递归的运算所有切片
	std::vector<std::string> slide_suffixs{ "sdpc", "srp", "svs", "mrxs" };
	std::vector<std::string> slide_dirs;
	std::vector<std::string> save_dirs;
	for (auto& p : std::filesystem::recursive_directory_iterator(slidePath)) {
		if (std::filesystem::is_directory(p)) {
			slide_dirs.emplace_back(p.path().string());
		}
	}
	for (auto dir : slide_dirs)
	{
		auto position = dir.find(slidePath);
		std::string dir_cp = dir;
		dir_cp.erase(position, slidePath.size());
		std::string temp_save_path = savePath + dir_cp;
		if (!std::filesystem::exists(temp_save_path)){
			std::filesystem::create_directories(temp_save_path);
		}
		std::vector<std::string> all_slides;
		for (auto slide_suffix : slide_suffixs)
		{
			std::vector<std::string> temp_slides = getFilesBySuffix(dir, slide_suffix);
			all_slides.insert(all_slides.end(), temp_slides.begin(), temp_slides.end());
		}
		std::vector<std::string> xml_list = getFilesBySuffix(temp_save_path, "xml"); 
		filterList(all_slides, xml_list);
		subTask(all_slides, slideProc, temp_save_path);
	}

	//vector<string> sdpcList;
	//vector<string> srpList;
	//vector<string> mrxsList;
	//vector<string> svsList;
	//getFiles(slidePath, sdpcList, "sdpc");
	//getFiles(slidePath, srpList, "srp");
	//getFiles(slidePath, svsList, "svs");
	//getFiles(slidePath, mrxsList, "mrxs");
	//vector<string> xmlList;
	//getFiles(savePath, xmlList, "xml");
	//filterList(sdpcList, xmlList);
	//filterList(srpList, xmlList);
	//filterList(svsList, xmlList);
	//filterList(mrxsList, xmlList);

	//vector<string> slideList;
	//slideList.insert(slideList.end(), sdpcList.begin(), sdpcList.end());
	//slideList.insert(slideList.end(), srpList.begin(), srpList.end());
	//slideList.insert(slideList.end(), mrxsList.begin(), mrxsList.end());
	//slideList.insert(slideList.end(), svsList.begin(), svsList.end());

	//subTask(slideList, slideProc, savePath);
}

//测试一下Caffe2Base.dll
//int main()
//{	
//	std::string config_path = "../x64/Release/config.ini";
//	setIniPath(config_path);
//	DetectDll detect_dll(config_path);
//	cv::Mat img = cv::imread("D:\\TEST_DATA\\1100037_1\\12_3.tif");
//	std::vector<cv::Mat> imgs;
//	imgs.emplace_back(img);
//	auto result = detect_dll.runModelInterface(imgs);
//	for (auto elem : result)
//	{
//		cout << elem.first << ": " << elem.second << endl;
//	}
//	system("pause");
//	return 0;
//}

//int main(int args, char* argv[])
//{
//	if (args > 2)
//	{
//		cout << "usage: exe IniPath" << endl;
//		return -1;
//	}
//	else if (args == 1)
//	{
//		_putenv_s("CUDA_VISIBLE_DEVICES", "0");
//		string iniPath = "./config.ini";
//		setIniPath(iniPath);
//		cout << IniConfig::instance().getIniString("TfModel1", "path") << endl;
//		DWORD dirType = GetFileAttributesA(iniPath.c_str());
//		if (dirType == INVALID_FILE_ATTRIBUTES) {
//			cout << "ini file doesn't exist!" << endl;
//			return -1;
//		}
//		startRun(iniPath.c_str());
//		system("pause");
//		return 0;
//	}
//	else
//	{
//		DWORD dirType = GetFileAttributesA(argv[1]);
//		if (dirType == INVALID_FILE_ATTRIBUTES) {
//			cout << "ini file doesn't exist!" << endl;
//			return -1;
//		}
//		setIniPath(argv[1]);
//		startRun(argv[1]);
//		return 0;
//	}
//	return 0;
//}
