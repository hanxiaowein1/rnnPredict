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
#include "Model3Holder.h"
/*
������һ����Ƭ�ļ�����̣�
�������ĳ�Ա�����У�
1.pbģ�ͺ�xgboostģ��
2.��Ƭ���ļ��ӿ�(MultiImageRead)(PS:Ҳ����ֱ��д��һ�������У���������ʱ�ͷ�)
*/

class SlideProc
{
private:
	Model1Holder *m1Holder = nullptr;
	Model2Holder* m2Holder = nullptr;
	Model3Holder* m3Holder = nullptr;
	vector<rnn*> rnnHandle;
	vector<regionResult> rResults;
	int recomNum = 30;
	string m_slide;
	SrpSlideRead* m_srpRead = nullptr;
	SdpcSlideRead* m_sdpcRead = nullptr;
	OpenSlideRead* m_osRead = nullptr;
	//Ƭ�ӵ���Ϣ(ÿ��ȥ���ʮ���鷳��������ֱ�ӳ�ʼ�����Ժ�õ���)
	int slideHeight;
	int slideWidth;
	double slideMpp;
	double slideScore;
	double slideRatio;
	//model1��model2�������Ϣ
	int model1Height;
	int model2Height;
	int model1Width;
	int model2Width;
	float model1Mpp;
	float model2Mpp;
	float model1OverlapRatio = 0.25f;

	std::condition_variable queue_cv1;
	std::condition_variable queue_cv2;
	std::condition_variable tensor_queue_cv2;
	int model1_batchsize = 20;
	int model2_batchsize = 30;
	std::mutex queue1Lock;
	std::mutex queue2Lock;
	std::mutex tensor_queue_lock2;
	std::atomic<bool> enterFlag1 = false;
	std::atomic<bool> enterFlag2 = false;

	std::atomic<bool> enterFlag7 = false;
	std::atomic<bool> enterFlag8 = false;
	std::atomic<bool> enterFlag9 = false;
	std::atomic<bool> enterFlag10 = false;
	queue<std::pair<cv::Rect, cv::Mat>> model1Queue;//��������Ӷ��̶߳�ͼ֮��Ȼ���ڽ�����ز�����ͼ��
	queue<std::pair<cv::Rect, cv::Mat>> model2Queue;
	queue<std::pair<vector<cv::Rect>, Tensor>> tensor_queue2;

	//��ȡ�Ŀ����Ϣ
	int block_height = 8192;
	int block_width = 8192;//�ڵ�0ͼ���ȡ��ͼ��Ĵ�С
	int read_level = 1;//model1��ȡ�Ĳ㼶
	int levelBin = 4;
	cv::Mat imgL4;
	cv::Mat binImg;
	int m_thre_col = 20;//rgb����ֵ(��mpp�޹�)
	int m_thre_vol = 150;//�������ֵ(ǰ���ָ�)
	int m_crop_sum;//��binImg��ͼ�������ֵ
	
private:
	//��ʼ��ģ��
	void initialize_handler(const char* iniPath);
	void model1Config(string model1Path);
	void model2Config(string model2Path);
	void rnnConfig(string rnnParentPath);
	void freeMemory();
	//������ʼ��model1��ȫͼ�еĿ��
	vector<cv::Rect> iniRects(MultiImageRead& mImgRead);
	//���ݸ������꣬�����м���Ҫ��ô���в�ȡ(���һ�������ǰƬ�����಻ͬ�����ǿ��ȡ�Ĵ�С��ͬ)
	vector<cv::Rect> iniRects(int sHeight, int sWidth, int height, int width);
	//��ȡ����Ϊ�����һ����Ҳ����ͬ���࣬�������ȫƬ�ж��ٲ�ȡ����
	vector<cv::Rect> iniRects(int sHeight, int sWidth, int height, int width, int overlap);
	//model1��ȫƬ�����8192*8192��ͼ��
	vector<cv::Rect> get_rects_slide();
	//��level1������в�ȡͼ���ʼ������Ұ��
	vector<cv::Rect> get_rects_slide(MultiImageRead& mImgRead);
	//���model1��һ�����еĲ�ͼ
	vector<cv::Rect> iniRects(int sHeight, int sWidth, int height, int width, int overlap, bool flag_right, bool flag_down);
	//�Ƽ�10������(ѡȡǰ10������д�뵽srp�ļ�����)
	vector<Anno> regionProposal(int recom);
	float runRnn(vector<Anno>& anno, MultiImageRead& mImgRead);
	float runRnnThread(int i, Tensor &inputTensor);
	//�ĳ���ǰ10��ǰ20��ǰ30��tensor�ķ���
	float runRnnThread2(int i, Tensor& inputTensor);
	void sortResultsByCoor(vector<regionResult>& results);
	void sortResultsByScore(vector<regionResult>& results);
	void sortResultsByScore(vector<PointScore>& pss);
	//������ֵȥ��(ǰ������ȼ��ߣ���������ȼ���)
	void filterBaseOnPoint(vector<PointScore>& PointScores, int threshold);

	//��mImgRead��ȡһ��512����д���
	void enterModel1Queue(MultiImageRead& mImgRead);
	void enterModel2Queue(MultiImageRead& mImgRead);
	//��mImgRead��ȡһ���������д���
	void enterModel1Queue2(MultiImageRead& mImgRead);
	bool popModel1Queue(vector<std::pair<cv::Rect, cv::Mat>>& rectMats);//�ӱ������ж�ͼ
	bool popModel1Queue(vector<std::pair<vector<cv::Rect>, Tensor>>& rectsTensors);
	bool checkFlags2();
	bool popModel2Queue(vector<std::pair<cv::Rect, cv::Mat>>& rectMats);
	bool popModel2Queue(vector<std::pair<vector<cv::Rect>, Tensor>>& rectsTensors);
	bool initialize_binImg();
	//��ʼ������mpp��ration�仯�����¸��ֱ仯�Ĳ���
	bool iniPara(const char* slide, MultiImageRead& mImgRead);
	//���ڳ����г���ȥ��m_srpRead��m_sdpcRead��m_osRead
	bool iniPara2(const char* slide, MultiImageRead& mImgRead);
	//�Ժ��õ���
	cv::Rect point2Rect(int x, int y, float radius, float diameter);
	cv::Point rect2Point(int x, int y, float radius);
	//������Ps����ȥ��
	void removeDuplicatePS(vector<PointScore>& pss1, vector<PointScore>& pss2, int threshold);
	void saveImages(vector<PointScore>& pss, int radius, string savePath);
	vector<PointScore> anno2PS(vector<Anno>& annos);
public:
	//����ini�ļ���ʼ��ģ��
	SlideProc(const char* iniPath);
	//�ͷŵ�ģ��
	~SlideProc();
	//����Ҫ��annos������ȥ�������������д��srp����
	//bool runSlide(const char* slide, vector<Anno>& annos);
	//��labʹ��
	//bool runSlide2(const char* slide);
	//�¼���model3
	bool runSlide3(const char* slide, string filename);
	//����һ����Ƭ����ʼ����(��xiaomingʹ��)
	//bool runSlide(const char*slide);
	//����model12�Ľ������xml��ʾ
	void saveResult(string savePath, string saveName);
	//����model2��ȥ��֮��Ľ���������Ӵ�С
	void saveResult2(string savePath, string saveName);
	void saveResult3(string savePath, string saveName);
	//void saveImg();
	float getSlideScore()
	{
		return slideScore;
	}
};



#endif // !_SLIDEPROC_H_