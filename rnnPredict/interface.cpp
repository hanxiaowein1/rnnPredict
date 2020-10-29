#include "interface.h"
#include "DLLManager.h"
#include "IniConfig.h"
#include "types.h"
#include "SlideProc.h"

RnnHandle initialize_handle(const char* iniPath)
{
	setIniPath(iniPath);
	SlideProc* slideProc = new SlideProc(iniPath);
	DLLManager* manager = new DLLManager();
	auto myHandle = new std::pair<SlideProc*, DLLManager*>[1];
	myHandle->first = slideProc;
	myHandle->second = manager;
	return (RnnHandle)myHandle;
}

bool slideProcess(RnnHandle myHandle, const char* slidePath, Anno* annos, int* len, double* wholeScore, UpdateProgressFunc callback)
{
	setProgressFun(callback);
	std::pair<SlideProc*, DLLManager*>* myHandle2 = (std::pair<SlideProc*, DLLManager*>*)myHandle;
	vector<Anno> annos_v;
	bool flag = myHandle2->first->runSlide(slidePath, annos_v, *len);
	for (int i = 0; i < *len; i++)
	{
		annos[i].id = i;
		annos[i].type = 0;
		annos[i].x = annos_v[i].x;
		annos[i].y = annos_v[i].y;
		annos[i].score = annos_v[i].score;
	}
	*wholeScore = myHandle2->first->getSlideScore();
	return true;
}

//bool slideProcess(handle myHandle, const char* slidePath, double* wholeScore)
//{
//	std::pair<SlideProc*, DLLManager*>* myHandle2 = (std::pair<SlideProc*, DLLManager*>*)myHandle;
//	bool flag = myHandle2->first->runSlide(slidePath);
//	*wholeScore = myHandle2->first->getSlideScore();
//	return flag;
//}

void freeModelMem(RnnHandle myHandle)
{
	std::pair<SlideProc*, DLLManager*>* myHandle2 = (std::pair<SlideProc*, DLLManager*>*)myHandle;
	delete myHandle2->first;
	delete myHandle2->second;
}

//int main()
//{
//	_putenv_s("CUDA_VISIBLE_DEVICES", "0");
//	string slidePath = "D:\\TEST_DATA\\rnnPredict\\052800092.srp";
//	string iniPath = "./config.ini";
//	handle myHandle = initialize_handle(iniPath.c_str());
//	double wholeScore;
//	int len = 300;
//	Anno* annos = new Anno[len];
//	slideProcess(myHandle, slidePath.c_str(), annos, &len, &wholeScore);
//	freeModelMem(myHandle);
//
//	//将这个出来的annos自己写到srp文件里面进行测试
//	SrpSlideRead *srpRead = new SrpSlideRead(slidePath.c_str());
//	srpRead->callCleanAnno();
//	srpRead->callWriteAnno(annos, len);
//	srpRead->callWriteParamDouble("score", wholeScore);
//	cout << "whole score is:" << wholeScore << endl;
//
//	system("pause");
//	return 0;
//}