#include <iostream>
#include <string>
#include <windows.h>

typedef struct {
	int id;
	int x;
	int y;
	int type;
	double score;
}Anno;

typedef unsigned long long* handle;

typedef void(*CallBack)(int stage, int w, int h, int currentIndex);

void myCallBack(int stage, int w, int h, int currentIndex)
{
	printf("from callback: %d, %d, %d, %d\n", stage, w, h, currentIndex);
}

typedef handle(*initialize_handle_func)(const char* iniPath);
typedef bool (*slideProcess_func)(handle myHandle, const char* slidePath, Anno* annos, int* len, double* wholeScore, CallBack callback);
typedef void (*freeModelMem_func)(handle myHandle);

int main(int argc, char** argv)
{
	_putenv_s("CUDA_VISIBLE_DEVICES", "0");
	auto dll_handle = LoadLibraryA("rnnPredict_20201202.dll");
	if (!dll_handle) {
		printf("load library failed,last error code is: %d\n", GetLastError());
		return -1;
	}
	auto init_func = (initialize_handle_func)GetProcAddress(dll_handle, "initialize_handle");
	auto process_func = (slideProcess_func)GetProcAddress(dll_handle, "slideProcess");
	auto close_func = (freeModelMem_func)GetProcAddress(dll_handle, "freeModelMem");

	//auto  ctx = init_func("D:\\HGProject\\AI-win\\AI-interface\\HGTAIProcess\\x64\Debug\\libcsh\\config.ini");
	auto ctx = init_func("./config.ini");
	//std::string file_name = "D:\\data\\test1127\\1912251023.srp";
	std::string file_name = "D:\\TEST_DATA\\Slide\\BD_Hard_Rec\\BD pos review 2th\\TG2325067.srp";
	Anno annos[10];
	int annos_len = 10;
	double res_score = 0.0;
	auto flag = process_func(ctx, file_name.c_str(), annos, &annos_len, &res_score, myCallBack);
	printf("flag: \b, res score is: %f\n", flag, res_score);

	close_func(ctx);

	system("pause");
	return 0;
}