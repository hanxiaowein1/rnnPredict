#include "progress_record.h"
#include <iostream>
#include <map>
using namespace std;

int stage = -1;
std::map<int, Record> records;
UpdateProgressFunc progress_fun = nullptr;
int global_slide_width = 0;
int global_slide_height = 0;

void setGlobalSlideWidth(int width)
{
	global_slide_width = width;
}

void setGlobalSlideHeight(int height)
{
	global_slide_height = height;
}

void setProgressFun(UpdateProgressFunc in_progress_fun)
{
	progress_fun = in_progress_fun;
}

void setStage(int in_stage, int total)
{
	stage = in_stage;
	records[stage].total = total;
	records[stage].progress = 0;
	//cout << "Temp Stage: " << in_stage << ", total:" << total << endl;
}

void addStep(int step)
{
	if (records.find(stage) != records.end())
	{
		records[stage].progress = records[stage].progress + step;
	}
	else
	{
		std::cerr << "stage has not been initialized!\n";
	}
	if (progress_fun != nullptr)
	{
		progress_fun(stage, global_slide_width, global_slide_height, records[stage].progress);
	}
	//cout << records[stage].progress << " ";
}