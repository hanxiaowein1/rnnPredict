#pragma once

#ifndef _SLIDEPROC_H_
#define _SLIDEPROC_H_
#include <ctime>
#include "MultiImageRead.h"
#include "SrpSlideRead.h"
#include "Model1Holder.h"
#include "Model2Holder.h"
#include "Model3Holder.h"
#include "RnnHolder.h"
/*
������һ����Ƭ�ļ�����̣�
�������ĳ�Ա�����У�
1.pbģ�ͺ�xgboostģ��
2.��Ƭ���ļ��ӿ�(MultiImageRead)(PS:Ҳ����ֱ��д��һ�������У���������ʱ�ͷ�)
*/

class SlideProc
{
public:
	cv::Mat imgL4;
	//Ƭ�ӵ���Ϣ(ÿ��ȥ���ʮ���鷳��������ֱ�ӳ�ʼ�����Ժ�õ���)
	int slideHeight;
	int slideWidth;
	double slideMpp;
private:
	std::unique_ptr<Model1Holder> m1Holder;
	std::unique_ptr<Model2Holder> m2Holder;
	std::unique_ptr<Model3Holder> m3Holder;
	std::unique_ptr<RnnHolder> rnnHolder;
	vector<regionResult> rResults;
	int recomNum = 30;
	string m_slide;
	//SrpSlideRead* m_srpRead = nullptr;
	//SdpcSlideRead* m_sdpcRead = nullptr;
	//OpenSlideRead* m_osRead = nullptr;
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
	queue<std::pair<vector<cv::Rect>, tensorflow::Tensor>> tensor_queue2;

	//��ȡ�Ŀ����Ϣ
	int block_height = 8192;
	int block_width = 8192;//�ڵ�0ͼ���ȡ��ͼ��Ĵ�С
	int read_level = 1;//model1��ȡ�Ĳ㼶
	int levelBin = 4;
	cv::Mat binImg;
	int m_thre_col = 20;//rgb����ֵ(��mpp�޹�)
	int m_thre_vol = 150;//�������ֵ(ǰ���ָ�)
	int m_crop_sum;//��binImg��ͼ�������ֵ
	
private:
	//��ʼ��ģ��
	void initialize_handler(const char* iniPath);
	void model1Config();
	void model2Config();
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
	float runRnn(vector<Anno>& anno, MultiImageRead& mImgRead);
	void getM2Imgs(std::vector<cv::Rect> rects, std::vector<cv::Mat>& imgs, MultiImageRead& mImgRead);
	void sortResultsByCoor(vector<regionResult>& results);
	void sortResultsByScore(vector<regionResult>& results);
	void sortResultsByScore(vector<PointScore>& pss);
	//������ֵȥ��(ǰ������ȼ��ߣ���������ȼ���)
	void filterBaseOnPoint(vector<PointScore>& PointScores, int threshold);

	//��mImgRead��ȡһ��512����д���
	void enterModel1Queue(MultiImageRead& mImgRead);
	//��mImgRead��ȡһ���������д���
	void enterModel1Queue2(MultiImageRead& mImgRead);
	bool popModel1Queue(vector<std::pair<cv::Rect, cv::Mat>>& rectMats);//�ӱ������ж�ͼ
	bool popModel1Queue(vector<std::pair<vector<cv::Rect>, tensorflow::Tensor>>& rectsTensors);
	bool checkFlags2();
	//��ʼ������mpp��ration�仯�����¸��ֱ仯�Ĳ���
	bool iniPara(const char* slide, MultiImageRead& mImgRead);
	//���ڳ����г���ȥ��m_srpRead��m_sdpcRead��m_osRead
	bool iniPara2(const char* slide, MultiImageRead& mImgRead);
	//�Ժ��õ���
	cv::Rect point2Rect(int x, int y, float radius, float diameter);
	cv::Point rect2Point(int x, int y, float radius);
	//������Ps����ȥ��
	void removeDuplicatePS(vector<PointScore>& pss1, vector<PointScore>& pss2, int threshold);
	void saveImages(vector<PointScore>& pss, int radius, string savePath, MultiImageRead &mImgRead);
	vector<PointScore> anno2PS(vector<Anno>& annos);
public:
	//����ini�ļ���ʼ��ģ��
	SlideProc(const char* iniPath);
	//�ͷŵ�ģ��
	~SlideProc();
	//���ݴ����len����ȷ����Ҫ�������ٸ�annos
	bool runSlide(const char* slide, vector<Anno>& annos, int len);
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

	//�Ƽ�recom������
	vector<Anno> regionProposal(int recom);
	//void saveImg();
	float getSlideScore()
	{
		return slideScore;
	}
};



#endif // !_SLIDEPROC_H_