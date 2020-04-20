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
	
	//因为这个里面既有类别，又有分数，所以可以用分数或者结构体存储
	vector<model3Result> resultOutput(const Tensor& tensor);

	vector<model3Result> model3Process(vector<cv::Mat>& imgs);
	
};

#endif