#pragma once
#ifndef _TYPES_H_
#define _TYPES_H_
#include "opencv2/opencv.hpp"
//保存着model1的结果
typedef unsigned long long* handle;
struct model1Result
{
	float score;//model1的分数
	std::vector<cv::Point> points;//定位点
};

struct model2Result
{
	float score;//model2的分数
	std::vector<float> tensor;//2048的向量
};

//一个512*512块的结果
struct regionResult
{
	cv::Point point;//全局坐标
	model1Result result;//model的结果
	std::vector<float> score2;//model2的结果
};

struct PointScore
{
	cv::Point point;
	float score;
};

//写到srp文件里的信息
typedef struct {
	int id;
	int x;
	int y;
	int type;
	double score;
}Anno;

struct model3Result
{
	enum Type { TYPICAL = 0, ATYPICAL, NPLUS };
	Type type;
	float scores[3]{0};
	void iniType()
	{
		if (scores[0] >= scores[1] && scores[0] >= scores[2])
		{
			type = TYPICAL;
		}
		if (scores[1] >= scores[0] && scores[1] >= scores[2])
		{
			type = ATYPICAL;
		}
		if (scores[2] >= scores[0] && scores[2] >= scores[1])
		{
			type = NPLUS;
		}
	}
};

#endif