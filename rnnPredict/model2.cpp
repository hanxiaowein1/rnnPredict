#include "model2.h"

model2::model2()
{ 
}

model2::~model2()
{
}

model2::model2(modelConfig config, char* buffer, int size) :model(config, buffer, size)
{

}

vector<float> model2::resultOutput(const Tensor & tensor)
{
	vector<float> scores;
	if (tensor.dims() != 2)
	{
		cout << "model2 output size should be two...\n";
		return scores;
	}
	auto scoreTensor = tensor.tensor<float, 2>();
	for (int i = 0; i < tensor.dim_size(0); i++)
	{
		float score = (scoreTensor(i, 0));
		scores.emplace_back(score);
	}
	return scores;
}

//points是在model2分辨率下的model1的定位点
void model2::dataInput(const vector<cv::Point> &points, const cv::Mat &inMat, vector<cv::Mat> &imgs)
{
	int rows = inMat.rows;
	int cols = inMat.cols;
	for (int i = 0; i < points.size(); i++)
	{
		cv::Point center = points[i];
		//计算在model2分辨率下的坐标
		int top = 0, bottom = 0, left = 0, right = 0;
		top = (center.y - 128) > 0 ? (center.y - 128) : 0;
		bottom = (center.y + 127) >= rows ? (rows - 1) : (center.y + 127);
		left = (center.x - 128) > 0 ? (center.x - 128) : 0;
		right = (center.x + 127) >= cols ? (cols - 1) : (center.x + 127);
		//这几个就是在原图中裁掉的四个点的坐标
		cv::Rect rectMat(left, top, right - left + 1, bottom - top + 1);
		Mat tmp = inMat(rectMat);
		//cout << "after getSubImg" << endl;
		//计算需要贴在256*256白图上的坐标
		int topStick = (center.y - 128) > 0 ? 0 : abs(center.y - 128);
		int leftStick = (center.x - 128) > 0 ? 0 : abs(center.x - 128);
		cv::Mat WhiteMat(256, 256, CV_8UC3, Scalar(0, 0, 0));
		tmp.copyTo(WhiteMat(Rect(leftStick, topStick, tmp.cols, tmp.rows)));//得到图像
																			//cout << "after copy to whiteMat" << endl;
																			//imgResize(&WhiteMat, &WhiteMat, 256, 256);
		imgs.emplace_back(WhiteMat);

	}
}

vector<float> model2::model2ProcessResizeInPb(std::vector<cv::Mat>& imgs)
{
	vector<float> scores;
	if (imgs.size() == 0)
		return scores;
	int start = 0;
	for (int i = 0; i < imgs.size(); i = i + batchsize)
	{
		auto iterBegin = imgs.begin() + start;
		auto iterEnd = imgs.end();
		if (iterBegin + batchsize > iterEnd)
		{
			vector<cv::Mat> tempImgs(iterBegin, iterEnd);
			vector<tensorflow::Tensor> tempTensors;
			output(tempImgs, tempTensors, 1);
			vector<float> tempScore = resultOutput(tempTensors[0]);
			scores.insert(scores.end(), tempScore.begin(), tempScore.end());
		}
		else
		{
			iterEnd = iterBegin + batchsize;
			vector<cv::Mat> tempImgs(iterBegin, iterEnd);
			vector<tensorflow::Tensor> tempTensors;
			output(tempImgs, tempTensors, 1);
			vector<float> tempScore = resultOutput(tempTensors[0]);
			scores.insert(scores.end(), tempScore.begin(), tempScore.end());
			start = i + batchsize;
		}
	}
	return scores;
}

vector<float> model2::model2Process(vector<Tensor>& tensors)
{
	vector<float> scores;
	if (tensors.size() == 0)
		return scores;
	for (int i = 0; i < tensors.size(); i++)
	{
		vector<Tensor> tempTensors;
		output(tensors[i], tempTensors);
		vector<float> tempScore = resultOutput(tempTensors[0]);
		scores.insert(scores.end(), tempScore.begin(), tempScore.end());
	}
	return scores;
}

vector<float> model2::model2Process(vector<cv::Mat>& imgs)
{
	vector<float> scores;
	if (imgs.size() == 0)
		return scores;
	if (imgs[0].cols != this->getModelHeight() || imgs[0].rows != this->getModelWidth())
	{
		for (int i = 0; i < imgs.size(); i++)
		{
			cv::resize(imgs[i], imgs[i], cv::Size(this->getModelWidth(), this->getModelHeight()));
		}
	}
	int start = 0;
	for (int i = 0; i < imgs.size(); i = i + batchsize)
	{
		auto iterBegin = imgs.begin() + start;
		auto iterEnd = imgs.end();
		if (iterBegin + batchsize > iterEnd)
		{
			vector<cv::Mat> tempImgs(iterBegin, iterEnd);
			vector<tensorflow::Tensor> tempTensors;
			output(tempImgs, tempTensors);
			vector<float> tempScore = resultOutput(tempTensors[0]);
			scores.insert(scores.end(), tempScore.begin(), tempScore.end());
		}
		else
		{
			iterEnd = iterBegin + batchsize;
			vector<cv::Mat> tempImgs(iterBegin, iterEnd);
			vector<tensorflow::Tensor> tempTensors;
			output(tempImgs, tempTensors);
			vector<float> tempScore = resultOutput(tempTensors[0]);
			scores.insert(scores.end(), tempScore.begin(), tempScore.end());
			start = i + batchsize;
		}
	}
	return scores;
}
void model2::model2Process(vector<cv::Mat>& imgs, vector<tensorflow::Tensor>& tensors)
{
	this->output(imgs, tensors);
}
/*
rects是model1的裁图边框
*/
vector<float> model2::model2Process(
	const vector<cv::Rect> &rects, const vector<model1Result> &results, const cv::Mat &inMat, vector<cv::Mat> &imgs
)
{
	vector<cv::Point> points;
	for (auto iter = results.begin(); iter != results.end(); iter++)
	{
		int place = iter - results.begin();
		if (iter->score > 0.5)
		{
			for (int i = 1; i < iter->points.size(); i++)
			{
				Point point;
				point.x = iter->points[i].x;
				point.y = iter->points[i].y;
				point.x = rects[place].x + point.x;
				point.y = rects[place].y + point.y;//这个是model1分辨率下的坐标，需要转为model2分辨率下的坐标
				point.x = point.x * (model1Resolution / model2Resolution);
				point.y = point.y * (model1Resolution / model2Resolution);
				points.emplace_back(point);
			}
		}
	}
	Mat dstMat;
	int srcHeight = inMat.rows;
	int srcWidth = inMat.cols;
	int dstHeight = srcHeight * float(srcResolution / model2Resolution);//图像变大
	int dstWidth = srcWidth * float(srcResolution / model2Resolution);
	cv::resize(inMat, dstMat, cv::Size(dstWidth, dstHeight));
	dataInput(points, dstMat, imgs);
	//为model2也设置一个batchsize
	vector<float> score;
	int start = 0;
	for (int i = 0; i < imgs.size(); i = i + batchsize)
	{
		auto iterBegin = imgs.begin() + start;
		auto iterEnd = imgs.end();
		if (iterBegin + batchsize > iterEnd)
		{
			vector<cv::Mat> tempImgs(iterBegin, iterEnd);
			vector<tensorflow::Tensor> tempTensors;
			output(tempImgs, tempTensors);
			vector<float> tempScore = resultOutput(tempTensors[0]);
			score.insert(score.end(), tempScore.begin(), tempScore.end());
		}
		else
		{
			iterEnd = iterBegin + batchsize;
			vector<cv::Mat> tempImgs(iterBegin, iterEnd);
			vector<tensorflow::Tensor> tempTensors;
			output(tempImgs, tempTensors);
			vector<float> tempScore = resultOutput(tempTensors[0]);
			score.insert(score.end(), tempScore.begin(), tempScore.end());
			start = i + batchsize;
		}
	}
	return score;
}