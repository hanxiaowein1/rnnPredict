#pragma once

#ifndef _AILAB_PROGRESS_H_
#define _AILAB_PROGRESS_H_

#include <atomic>
struct Record {
	std::atomic<int> progress = 0;
	int total = 0;
};

typedef void(*UpdateProgressFunc)(int stage, int w, int h, int currentIndex);

void setProgressFun(UpdateProgressFunc in_progress_fun);
void setGlobalSlideWidth(int width);
void setGlobalSlideHeight(int height);
void setStage(int in_stage, int total);
void addStep(int step);

#endif