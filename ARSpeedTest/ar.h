#pragma once
#ifndef _AILAB_AR_H_
#define _AILAB_AR_H_

//还是按照原始的方法，一个handle
#include "model_holder.h"
#include "anno.h"
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"

#ifdef _ARFUNC_EXPORT_
#define AR_API extern "C" __declspec(dllexport)
#else 
#define AR_API extern "C" __declspec(dllimport)
#endif

struct ARConfig {
	double img_mpp = 0.293f;
	std::string img_path = "";
	std::string img_save_path = "";
	std::string img_suffix = "";
	int max_recom_num = 10;
	std::string cuda_visible_devices = "0";
	double remove_threshold = 50.0f;
	double score_threshold = 0.5f;
	void showARConfig()
	{
		std::cout << "img_mpp:" << img_mpp << ", img_path:" << img_path << ", img_save_path:" << img_save_path
			<< ", max_recom_num:" << max_recom_num << ", cuda_visible_devices:" << cuda_visible_devices << std::endl;
	}
};

AR_API ArHandle initialize_handle(std::string ini_path);
//AR_API bool slideProcess(ArHandle myHandle, const char* slidePath, Anno* annos, int* len, double* wholeScore, UpdateProgressFunc callback);
AR_API void process(
	std::vector<Anno>& annos, ArHandle myHandle, cv::Mat& raw_img, 
	double img_mpp, int n, double score_threshold, double remove_threshold);
AR_API void freeModelMem(ArHandle myHandle);

#endif