#pragma once
#ifndef _SLIDEFACTORY_H_
#define _SLIDEFACTORY_H_
#include <string>
#include "SlideRead.h"
#include "OpenSlideRead.h"
#include "SdpcSlideRead.h"
#include "SrpSlideRead.h"
#include "commonFunction.h"
class SlideFactory
{
public:
	SlideFactory();
	~SlideFactory();
public:
	unique_ptr<SlideRead> createSlideProduct(const char* slidePath);
};

#endif