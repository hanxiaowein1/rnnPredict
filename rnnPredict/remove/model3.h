#pragma once
#ifndef _MODEL3_H_
#define _MODEL3_H_
#include "model.h"
#include "types.h"
class model3 : public model
{
private:
	const float model3Resolution = 0.293f;
	int batchsize = 10;
public:
	model3();
	model3(modelConfig config, char* buffer, int size);
	~model3() {};
	
	//��Ϊ����������������з��������Կ����÷������߽ṹ��洢
	vector<model3Result> resultOutput(const Tensor& tensor);

	vector<model3Result> model3Process(vector<cv::Mat>& imgs);
	
};

#endif