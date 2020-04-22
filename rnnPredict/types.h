#pragma once
#ifndef _TYPES_H_
#define _TYPES_H_
#include "opencv2/opencv.hpp"
//������model1�Ľ��
typedef unsigned long long* handle;
struct model1Result
{
	float score;//model1�ķ���
	std::vector<cv::Point> points;//��λ��
};

struct model2Result
{
	float score;//model2�ķ���
	std::vector<float> tensor;//2048������
};

//һ��512*512��Ľ��
struct regionResult
{
	cv::Point point;//ȫ������
	model1Result result;//model�Ľ��
	std::vector<float> score2;//model2�Ľ��
};

struct PointScore
{
	cv::Point point;
	float score;
};

//д��srp�ļ������Ϣ
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