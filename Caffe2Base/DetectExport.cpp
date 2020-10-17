#include "DetectExport.h"
#include "IniConfig.h"

//class DetectModelDeleter
//{
//public:
//	void operator()(DetectHandle *handle)
//	{
//		DetectModel* detect_model = (DetectModel*)handle->handle;
//		delete detect_model;
//	}
//};

DetectHandle getDetectHandle(std::string config_path)
{
	setIniPath(config_path);
	DetectModel* detect_model = new DetectModel("DetectModel");
	detect_model->createThreadPool(1);
	//return (DetectHandle)detect_model;
	return reinterpret_cast<DetectHandle>(detect_model);
}

DetectResult runModel(DetectHandle handle, std::vector<cv::Mat>& imgs)
{
	//在里面将其转化为DetectModel
	DetectModel* detect_model = reinterpret_cast<DetectModel*>(handle);
	detect_model->processDataConcurrency(imgs);
	DetectResult detect_result = detect_model->m_result;
	return detect_result;
}

void releaseDetectHandle(DetectHandle handle)
{
	DetectModel* detect_model = reinterpret_cast<DetectModel*>(handle);
	delete detect_model;
}