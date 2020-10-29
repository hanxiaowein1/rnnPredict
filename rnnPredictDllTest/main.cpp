#include "interface.h"
#include "anno.h"
#include <iostream>
#include <string>
#include <windows.h>

//UpdateProgressFunc 

void testProcess(int stage, int w, int h, int currentIndex)
{
	
}

int main()
{
	_putenv_s("CUDA_VISIBLE_DEVICES", "0");
	std::string config_path = "D:\\Programmer\\C++\\AILabProject\\Solution2\\rnnPredict\\x64\\Release\\config.ini";
	std::string slide_path = "D:\\TEST_DATA\\svs\\new_svs\\20200436-asc-h.svs";
	RnnHandle rnn_handle = initialize_handle(config_path.c_str());
	Anno annos[10];
	int len = 10;
	double wholeScore = 0.0f;
	slideProcess(rnn_handle, slide_path.c_str(), annos, &len, &wholeScore, testProcess);
	freeModelMem(rnn_handle);
	for (int i = 0; i < len; i++)
	{
		std::cout << annos[i].x << ", " << annos[i].y << ", " << annos[i].score << std::endl;
	}
	system("pause");
	return 0;
}