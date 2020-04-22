#include <vector>
#include <string>
#include <sstream>
#include "opencv2/opencv.hpp"

//分解string字符串
std::vector<std::string> split(std::string& s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<cv::Point> getRegionPoints2(cv::Mat& mask, float threshold)
{
	//cout << "enter getRegionPoints2" <<endl;
	//先直接进行筛选操作
	double minVal;
	double maxVal;
	cv::Point minLoc;
	cv::Point maxLoc;
	minMaxLoc(mask, &minVal, &maxVal, &minLoc, &maxLoc);
	//cout << "maxVal:" << maxVal << endl;
	//对图像进行过滤，大于阈值的等于原图像
	cv::threshold(mask, mask, threshold * maxVal, maxVal, cv::THRESH_TOZERO);
	//cout << "after thresHold ,the mask is" << *mask << endl;
	//归一化到0-255
	cv::Mat matForConn = mask.clone();
	cv::normalize(matForConn, matForConn, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	//cout << "after normalize ,the mask is" <<endl<< *mask << endl;
	//寻找连通域的lable图
	cv::Mat labels;
	//conn知道到底有几个连通域，其中0代表的是背景，1-(conn-1)，则是前景的部分
	int conn = cv::connectedComponents(matForConn, labels, 8, CV_32S);
	//cout << "the lables is:"<<endl << labels << endl;
	//求每个连通域最大值的坐标，若有多个最大值，取第一个最大值
	std::vector<float> maxValueConn(conn, 0);//保存每个连通域的最大值
	std::vector<cv::Point> points(conn, cv::Point(0, 0));

	for (int i = 0; i < labels.rows; i++) {
		int* LinePtr = (int*)labels.ptr(i);
		float* LinePtrMask = (float*)mask.ptr(i);
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
