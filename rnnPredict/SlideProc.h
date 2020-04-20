#pragma once

#ifndef _SLIDEPROC_H_
#define _SLIDEPROC_H_
#include <ctime>
#include "model1.h"
#include "model2.h"
#include "model3.h"
#include "rnn.h"
#include "MultiImageRead.h"
#include "SrpSlideRead.h"
#include "Model1Holder.h"
#include "Model2Holder.h"

/*
管理者一张切片的计算过程：
因此所需的成员变量有：
1.pb模型和xgboost模型
2.切片的文件接口(MultiImageRead)(PS:也可以直接写在一个函数中，函数结束时释放)
*/

class SlideProc
{
private:
	//xgdll的相关函数
	HINSTANCE xgDll = nullptr;
	typedef handle(*function_initialize)(string);
	typedef void(*function_free)(handle);
	typedef float(*function_getPredictValue)(std::vector<float>&, std::vector<float>&, handle);//得到预测值
	function_initialize initialize_xgboost = nullptr;
	function_getPredictValue getPredictValue = nullptr;
	function_free free_xgboost = nullptr;

	Model1Holder *m1Holder = nullptr;
	Model2Holder* m2Holder = nullptr;
	model1 *model1Handle = nullptr;
	model2* model2Handle = nullptr;
	model3* model3Handle = nullptr;
	vector<rnn*> rnnHandle;
	vector<handle> xgHandle;
	vector<regionResult> rResults;
	int recomNum = 30;
	string m_slide;
	SrpSlideRead* m_srpRead = nullptr;
	SdpcSlideRead* m_sdpcRead = nullptr;
	OpenSlideRead* m_osRead = nullptr;
	//片子的信息(每次去获得十分麻烦，不如先直接初始化，以后好调用)
	int slideHeight;
	int slideWidth;
	double slideMpp;
	double slideScore;
	double slideRatio;
	//model1和model2的相关信息
	int model1Height;
	int model2Height;
	int model1Width;
	int model2Width;
	float model1Mpp;
	float model2Mpp;
	float model1OverlapRatio = 0.25f;

	std::condition_variable queue_cv1;
	std::condition_variable queue_cv2;
	std::condition_variable tensor_queue_cv;
	std::condition_variable tensor_queue_cv2;
	int model1_batchsize = 20;
	int model2_batchsize = 30;
	std::mutex queue1Lock;
	std::mutex queue2Lock;
	std::mutex tensor_queue_lock;
	std::mutex tensor_queue_lock2;
	std::atomic<bool> enterFlag1 = false;
	std::atomic<bool> enterFlag2 = false;
	/*std::vector<std::atomic<bool>> enterFlag3;*/
	std::atomic<bool> enterFlag3 = false;
	std::atomic<bool> enterFlag4 = false;
	std::atomic<bool> enterFlag5 = false;
	std::atomic<bool> enterFlag6 = false;

	std::atomic<bool> enterFlag7 = false;
	std::atomic<bool> enterFlag8 = false;
	std::atomic<bool> enterFlag9 = false;
	std::atomic<bool> enterFlag10 = false;
	queue<std::pair<cv::Rect, cv::Mat>> model1Queue;//用来保存从多线程读图之后然后在进行相关操作的图像
	queue<std::pair<cv::Rect, cv::Mat>> model2Queue;
	queue<std::pair<vector<cv::Rect>, Tensor>> tensor_queue;
	queue<std::pair<vector<cv::Rect>, Tensor>> tensor_queue2;

	//裁取的宽高信息
	int block_height = 8192;
	int block_width = 8192;//在第0图层读取的图像的大小
	int read_level = 1;//model1读取的层级
	int levelBin = 4;
	cv::Mat imgL4;
	cv::Mat binImg;
	int m_thre_col = 20;//rgb的阈值(与mpp无关)
	int m_thre_vol = 150;//面积的阈值(前景分割)
	int m_crop_sum;//从binImg抠图的求和阈值
	
private:
	//初始化模型
	void initialize_handler(const char* iniPath);
	void model1Config(string model1Path);
	void model2Config(string model2Path);
	void model3Config(string model3Path);
	void rnnConfig(string rnnParentPath);
	void loadXgdll();
	void xgConfig(string xgParentPath);
	void freeMemory();
	//用来初始化model1在全图中的框框
	vector<cv::Rect> iniRects(MultiImageRead& mImgRead);
	//根据给的坐标，来进行计算要怎么进行裁取(最后一块冗余和前片块冗余不同，但是块裁取的大小相同)
	vector<cv::Rect> iniRects(int sHeight, int sWidth, int height, int width);
	//裁取策略为，最后一个块也是相同冗余，而且最后全片有多少裁取多少
	vector<cv::Rect> iniRects(int sHeight, int sWidth, int height, int width, int overlap);
	//model1在全片上面裁8192*8192的图像
	vector<cv::Rect> get_rects_slide();
	//在level1上面进行采取图像初始化的视野块
	vector<cv::Rect> get_rects_slide(MultiImageRead& mImgRead);
	//针对model1在一个块中的裁图
	vector<cv::Rect> iniRects(int sHeight, int sWidth, int height, int width, int overlap, bool flag_right, bool flag_down);
	void runModel1(MultiImageRead &mImgRead);
	void runModel2(MultiImageRead& mImgRead);
	//gxb和mjb的新模型
	vector<PointScore> runModel3(MultiImageRead& mImgRead);
	//model3的推荐策略
	vector<PointScore> model3Recom(vector<std::pair<cv::Rect, model3Result>>& xyResults);
	//推荐10个区域(选取前10个区域写入到srp文件里面)
	vector<Anno> regionProposal(int recom);
	float runRnn(vector<Anno>& anno, MultiImageRead& mImgRead);
	float runRnnThread(int i, Tensor &inputTensor);
	//改成了前10，前20，前30的tensor的分数
	float runRnnThread2(int i, Tensor& inputTensor);
	float runXgboost();
	void sortResultsByCoor(vector<regionResult>& results);
	void sortResultsByScore(vector<regionResult>& results);
	void sortResultsByScore(vector<PointScore>& pss);
	//根据阈值去重(前面的优先级高，后面的优先级低)
	void filterBaseOnPoint(vector<PointScore>& PointScores, int threshold);

	//从mImgRead读取一个512块进行处理
	void enterModel1Queue(MultiImageRead& mImgRead);
	void enterModel2Queue(MultiImageRead& mImgRead);
	void enterModel2Queue2(std::atomic<bool>& flag, MultiImageRead& mImgRead);
	//从mImgRead读取一个长条进行处理
	void enterModel1Queue2(MultiImageRead& mImgRead);
	//读取长条之后转为tensor
	void enterModel1Queue3(std::atomic<bool> &flag, MultiImageRead& mImgRead);
	//读取8192*8192的图之后转为tensor
	void enterModel1Queue4(std::atomic<bool>& flag, MultiImageRead& mImgRead);
	//以batchsize为阈值，将imgs放到tensors里面，每batchsize的图像放到一个tensors里面
	void Mats2Tensors(vector<cv::Mat> &imgs, vector<Tensor> &tensors, int batchsize);
	void Mats2Tensors(vector<std::pair<cv::Rect, cv::Mat>>& rectMats, vector<std::pair<vector<cv::Rect>, Tensor>> & rectsTensors, int batchsize);
	void normalize(vector<cv::Mat>& imgs, Tensor& tensor);
	//model2无需从全片上扣取一个长条
	//void enterModel2Queue2(MultiImageRead& mImgRead);
	bool popModel1Queue(vector<std::pair<cv::Rect, cv::Mat>>& rectMats);//从本队列中读图
	bool popModel1Queue(vector<std::pair<vector<cv::Rect>, Tensor>>& rectsTensors);
	bool checkFlags();
	bool checkFlags2();
	bool popModel2Queue(vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	bool popModel2Queue(vector<std::pair<vector<cv::Rect>, Tensor>>& rectsTensors);
	//关于图像前景分割（为了去除不用计算的部分）
	void threshold_segmentation(cv::Mat& img, cv::Mat& binImg, int level, int thre_col, int thre_vol);
	void remove_small_objects(cv::Mat& binImg, int thre_vol);
	bool initialize_binImg();
	//初始化根据mpp和ration变化而导致各种变化的参数
	bool iniPara(const char* slide, MultiImageRead& mImgRead);
	//先在程序中尝试去掉m_srpRead，m_sdpcRead，m_osRead
	bool iniPara2(const char* slide, MultiImageRead& mImgRead);
	//以后用得着
	cv::Rect point2Rect(int x, int y, float radius, float diameter);
	cv::Point rect2Point(int x, int y, float radius);
	//对两个Ps集合去重
	void removeDuplicatePS(vector<PointScore>& pss1, vector<PointScore>& pss2, int threshold);
	void saveImages(vector<PointScore>& pss, int radius, string savePath);
	void saveImages(MultiImageRead& mImgRead, vector<cv::Rect>& rects);
	vector<PointScore> anno2PS(vector<Anno>& annos);
public:
	//根据ini文件初始化模型
	SlideProc(const char* iniPath);
	//释放掉模型
	~SlideProc();
	//华乐要求将annos给传出去，而不是我这边写到srp里面
	bool runSlide(const char* slide, vector<Anno>& annos);
	//给lab使用
	bool runSlide2(const char* slide);
	//新加入model3
	bool runSlide3(const char* slide, string filename);
	//传入一张切片，开始计算(给xiaoming使用)
	bool runSlide(const char*slide);
	//保存model12的结果，用xml表示
	void saveResult(string savePath, string saveName);
	//保存model2的去重之后的结果，分数从大到小
	void saveResult2(string savePath, string saveName);
	void saveResult3(string savePath, string saveName);
	//void saveImg();
	float getSlideScore()
	{
		return slideScore;
	}
};



#endif // !_SLIDEPROC_H_