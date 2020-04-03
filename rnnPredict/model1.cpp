#include "model1.h"

model1::model1()
{
}

model1::~model1()
{
}

model1::model1(modelConfig config, char* buffer, int size) :model(config, buffer, size)
{
}

vector<model1Result> model1::resultOutput(vector<tensorflow::Tensor> &tensors)
{
	vector<model1Result> retResults;
	if (tensors.size() != 2)
	{
		cout << "model1Base::output: tensors size should be 2\n";
		return retResults;
	}
	auto scores = tensors[0].tensor<float, 2>();
	for (int i = 0; i < tensors[0].dim_size(0); i++)
	{
		model1Result result;
		Mat dst2;
		TensorToMat(tensors[1].Slice(i, i + 1), &dst2);
		//cv::imwrite("D:\\TEST_OUTPUT\\rnnPredict\\mask" + to_string(i) + ".tif", dst2);
		result.points = getRegionPoints2(&dst2, 0.7);
		result.score = scores(i, 0);
		retResults.emplace_back(result);
	}
	return retResults;
}


void model1::TensorToMat(Tensor mask, Mat *dst)
{
	float *data = new float[(mask.dim_size(1))*(mask.dim_size(2))];
	auto output_c = mask.tensor<float, 4>();
	//cout << "data 1 :" << endl;
	for (int j = 0; j < mask.dim_size(1); j++) {
		for (int k = 0; k < mask.dim_size(2); k++) {
			data[j * mask.dim_size(1) + k] = output_c(0, j, k, 1);
		}
	}
	Mat myMat = Mat(mask.dim_size(1), mask.dim_size(2), CV_32FC1, data);
	//cout << myMat;
	*dst = myMat.clone();
	delete[]data;
}

vector<cv::Point> model1::getRegionPoints2(Mat *mask, float threshold)
{
	//cout << "enter getRegionPoints2" <<endl;
	//先直接进行筛选操作
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	minMaxLoc(*mask, &minVal, &maxVal, &minLoc, &maxLoc);
	//cout << "maxVal:" << maxVal << endl;
	//对图像进行过滤，大于阈值的等于原图像
	cv::threshold(*mask, *mask, threshold*maxVal, maxVal, THRESH_TOZERO);
	//cout << "after thresHold ,the mask is" << *mask << endl;
	//归一化到0-255
	Mat matForConn = (*mask).clone();
	cv::normalize(matForConn, matForConn, 0, 255, NORM_MINMAX, CV_8UC1);
	//cout << "after normalize ,the mask is" <<endl<< *mask << endl;
	//寻找连通域的lable图
	cv::Mat labels;
	//conn知道到底有几个连通域，其中0代表的是背景，1-(conn-1)，则是前景的部分
	int conn = cv::connectedComponents(matForConn, labels, 8, CV_32S);
	//cout << "the lables is:"<<endl << labels << endl;
	//求每个连通域最大值的坐标，若有多个最大值，取第一个最大值
	vector<float> maxValueConn(conn, 0);//保存每个连通域的最大值
	vector<cv::Point> points(conn, cv::Point(0, 0));

	for (int i = 0; i < labels.rows; i++) {
		int *LinePtr = (int*)labels.ptr(i);
		float* LinePtrMask = (float*)(*mask).ptr(i);
		for (int j = 0; j < labels.cols; j++) {
			//查看这个点属于哪一个连通域(1-(conn-1))
			int label = *(LinePtr + j);
			if (label == 0) {
				continue;
			}
			float value = *(LinePtrMask + j);
			//只有大于的时候，才会记录，等于的时候，不保存，为了避免以后会有重复的最大值，只取第一个最大值
			if (value > maxValueConn[label]) {
				maxValueConn[label] = value;//保留最大值
				points[label].x = j;//保留最大值的下标
				points[label].y = i;
			}
		}
	}
	//还有将points转为512*512中的点
	for (int i = 0; i < points.size(); i++) {
		points[i].x = int((points[i].x + 0.5) * (512 / 16));
		points[i].y = int((points[i].y + 0.5) * (512 / 16));
	}
	return points;//记住，第一个点不代表什么东西
}


void model1::iniRects(int height, int width, int sHeight, int sWidth)
{
	m_rects.clear();
	if (height == 0 || width == 0 || sHeight == 0 || sWidth == 0)
	{
		cout << "iniRects: some parameters should not be zero\n";
		return;
	}
	if (height < sHeight || width < sWidth)
	{
		cout << "size to be cropped should bigger \n";
		return;
	}
	int xNum = 0;//水平方向的裁剪个数
	int yNum = 0;//垂直方向的裁剪个数
	int overlap = 222;
	if (sHeight <= overlap || sWidth <= overlap)
	{
		cout << "sHeight and sWidth seems to small\n";
		return;
	}
	yNum = 1 + (height - sHeight) / (sHeight - overlap);
	xNum = 1 + (width - sWidth) / (sWidth - overlap);
	vector<int> xStart;
	vector<int> yStart;
	for (int i = 0; i < xNum; i++)
	{
		xStart.emplace_back((sWidth - overlap)*i);
	}
	for (int i = 0; i < yNum; i++)
	{
		yStart.emplace_back((sHeight - overlap)*i);
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
			m_rects.emplace_back(rect);
		}
	}
}

vector<model1Result> model1::model1Process(std::vector<Tensor>& tensorInput)
{
	vector<model1Result> results;
	if (tensorInput.size() == 0)
		return results;
	for (int i = 0; i < tensorInput.size(); i++)
	{
		vector<Tensor> tempTensors;
		output(tensorInput[i], tempTensors);
		vector<model1Result> tempResults = resultOutput(tempTensors);
		results.insert(results.end(), tempResults.begin(), tempResults.end());
	}
	return results;
}

vector<model1Result> model1::model1ProcessResizeInPb(std::vector<cv::Mat>& imgs)
{
	vector<model1Result> results;
	if (imgs.size() == 0)
		return results;
	//resize之后，送入模型进行计算
	int start = 0;
	for (int i = 0; i < imgs.size(); i = i + batchsize)
	{
		auto iterBegin = imgs.begin() + start;
		vector<cv::Mat>::iterator iterEnd = imgs.end();
		if (iterBegin + batchsize >= iterEnd)
		{
			vector<cv::Mat> tempImgs(iterBegin, iterEnd);
			vector<tensorflow::Tensor> tempTensors;
			output(tempImgs, tempTensors, 1);
			vector<model1Result> tempResults = resultOutput(tempTensors);
			results.insert(results.end(), tempResults.begin(), tempResults.end());
		}
		else
		{
			iterEnd = iterBegin + batchsize;
			vector<cv::Mat> tempImgs(iterBegin, iterEnd);
			vector<tensorflow::Tensor> tempTensors;
			output(tempImgs, tempTensors, 1);
			vector<model1Result> tempResults = resultOutput(tempTensors);
			results.insert(results.end(), tempResults.begin(), tempResults.end());
			start = i + batchsize;
		}

	}
	return results;
}

vector<model1Result> model1::model1Process(std::vector<cv::Mat>& imgs)
{
	vector<model1Result> results;
	if (imgs.size() == 0)
		return results;
	//如果传入的imgs的size不为512，那么就进行resize
	if (imgs[0].cols != this->getModelHeight() || imgs[0].rows != this->getModelWidth())
	{
		for (int i = 0; i < imgs.size(); i++)
		{
			cv::resize(imgs[i], imgs[i], cv::Size(this->getModelWidth(), this->getModelHeight()));
		}
	}
	//resize之后，送入模型进行计算
	int start = 0;
	for (int i = 0; i < imgs.size(); i = i + batchsize)
	{
		auto iterBegin = imgs.begin() + start;
		vector<cv::Mat>::iterator iterEnd = imgs.end();
		if (iterBegin + batchsize >= iterEnd)
		{
			vector<cv::Mat> tempImgs(iterBegin, iterEnd);
			vector<tensorflow::Tensor> tempTensors;
			output(tempImgs, tempTensors);
			vector<model1Result> tempResults = resultOutput(tempTensors);
			results.insert(results.end(), tempResults.begin(), tempResults.end());
		}
		else
		{
			iterEnd = iterBegin + batchsize;
			vector<cv::Mat> tempImgs(iterBegin, iterEnd);
			vector<tensorflow::Tensor> tempTensors;
			output(tempImgs, tempTensors);
			vector<model1Result> tempResults = resultOutput(tempTensors);
			results.insert(results.end(), tempResults.begin(), tempResults.end());
			start = i + batchsize;
		}

	}
	return results;
}

vector<model1Result> model1::model1Process(const cv::Mat &inMat)
{
	Mat dstMat;
	int srcHeight = inMat.rows;
	int srcWidth = inMat.cols;
	int dstHeight = srcHeight * float(srcResolution / model1Resolution);//图像变小
	int dstWidth = srcWidth * float(srcResolution / model1Resolution);
	cv::resize(inMat, dstMat, cv::Size(dstWidth, dstHeight));
	iniRects(dstHeight, dstWidth, 512, 512);
	//裁进入model1的小图
	//vector<cv::Rect> rects = getRects(dstHeight, dstWidth, 512, 512);
	vector<cv::Mat> imgs;
	for (int i = 0; i < m_rects.size(); i++)
	{
		imgs.emplace_back(dstMat(m_rects[i]));
	}
	int batchsize = 15;
	int start = 0;
	vector<model1Result> results;
	for (int i = 0; i < imgs.size(); i = i + batchsize)
	{
		auto iterBegin = imgs.begin() + start;
		vector<cv::Mat>::iterator iterEnd = imgs.end();
		if (iterBegin + batchsize >= iterEnd)
		{
			vector<cv::Mat> tempImgs(iterBegin, iterEnd);
			vector<tensorflow::Tensor> tempTensors;
			output(tempImgs, tempTensors);
			vector<model1Result> tempResults = resultOutput(tempTensors);
			results.insert(results.end(), tempResults.begin(), tempResults.end());
		}
		else
		{
			iterEnd = iterBegin + batchsize;
			vector<cv::Mat> tempImgs(iterBegin, iterEnd);
			vector<tensorflow::Tensor> tempTensors;
			output(tempImgs, tempTensors);
			vector<model1Result> tempResults = resultOutput(tempTensors);
			results.insert(results.end(), tempResults.begin(), tempResults.end());
			start = i + batchsize;
		}
		
	}
	
	return results;
}

void model1::convertPointM1ToSrc(Point &point)
{
	point.x = point.x*float(model1Resolution / srcResolution);//坐标变大
	point.y = point.y*float(model1Resolution / srcResolution);
}