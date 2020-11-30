#pragma once
#ifndef _AILAB_MODEL1_H_
#define _AILAB_MODEL1_H_

#include <vector>
#include <string>
#include <sstream>
#include "opencv2/opencv.hpp"

std::vector<cv::Point> getRegionPoints2(cv::Mat& mask, float threshold);

#endif