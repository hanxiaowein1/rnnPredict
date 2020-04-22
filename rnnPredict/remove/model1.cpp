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
	//��ֱ�ӽ���ɸѡ����
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	minMaxLoc(*mask, &minVal, &maxVal, &minLoc, &maxLoc);
	//cout << "maxVal:" << maxVal << endl;
	//��ͼ����й��ˣ�������ֵ�ĵ���ԭͼ��
	cv::threshold(*mask, *mask, threshold*maxVal, maxVal, THRESH_TOZERO);
	//cout << "after thresHold ,the mask is" << *mask << endl;
	//��һ����0-255
	Mat matForConn = (*mask).clone();
	cv::normalize(matForConn, matForConn, 0, 255, NORM_MINMAX, CV_8UC1);
	//cout << "after normalize ,the mask is" <<endl<< *mask << endl;
	//Ѱ����ͨ���lableͼ
	cv::Mat labels;
	//conn֪�������м�����ͨ������0������Ǳ�����1-(conn-1)������ǰ���Ĳ���
	int conn = cv::connectedComponents(matForConn, labels, 8, CV_32S);
	//cout << "the lables is:"<<endl << labels << endl;
	//��ÿ����ͨ�����ֵ�����꣬���ж�����ֵ��ȡ��һ�����ֵ
	vector<float> maxValueConn(conn, 0);//����ÿ����ͨ������ֵ
	vector<cv::Point> points(conn, cv::Point(0, 0));

	for (int i = 0; i < labels.rows; i++) {
		int *LinePtr = (int*)labels.ptr(i);
		float* LinePtrMask = (float*)(*mask).ptr(i);
		for (int j = 0; j < labels.cols; j++) {
			//�鿴�����������һ����ͨ��(1-(conn-1))
			int label = *(LinePtr + j);
			if (label == 0) {
				continue;
			}
			float value = *(LinePtrMask + j);
			//ֻ�д��ڵ�ʱ�򣬲Ż��¼�����ڵ�ʱ�򣬲����棬Ϊ�˱����Ժ�����ظ������ֵ��ֻȡ��һ�����ֵ
			if (value > maxValueConn[label]) {
				maxValueConn[label] = value;//�������ֵ
				points[label].x = j;//�������ֵ���±�
				points[label].y = i;
			}
		}
	}
	//���н�pointsתΪ512*512�еĵ�
	for (int i = 0; i < points.size(); i++) {
		points[i].x = int((points[i].x + 0.5) * (512 / 16));
		points[i].y = int((points[i].y + 0.5) * (512 / 16));
	}
	return points;//��ס����һ���㲻����ʲô����
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
	int xNum = 0;//ˮƽ����Ĳü�����
	int yNum = 0;//��ֱ����Ĳü�����
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
	//resize֮������ģ�ͽ��м���
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
	//��������imgs��size��Ϊ512����ô�ͽ���resize
	if (imgs[0].cols != this->getModelHeight() || imgs[0].rows != this->getModelWidth())
	{
		for (int i = 0; i < imgs.size(); i++)
		{
			cv::resize(imgs[i], imgs[i], cv::Size(this->getModelWidth(), this->getModelHeight()));
		}
	}
	//resize֮������ģ�ͽ��м���
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
	int dstHeight = srcHeight * float(srcResolution / model1Resolution);//ͼ���С
	int dstWidth = srcWidth * float(srcResolution / model1Resolution);
	cv::resize(inMat, dstMat, cv::Size(dstWidth, dstHeight));
	iniRects(dstHeight, dstWidth, 512, 512);
	//�ý���model1��Сͼ
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
	point.x = point.x*float(model1Resolution / srcResolution);//������
	point.y = point.y*float(model1Resolution / srcResolution);
}