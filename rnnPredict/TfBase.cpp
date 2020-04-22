#include "TfBase.h"

TfBase::TfBase(std::string iniPath, std::string group)
{
	fileProp.initByiniFile(iniPath, group);
	construct();
}

void TfBase::construct()
{
	tensorflow::GraphDef graph_def;
	tensorflow::Status load_graph_status =
		ReadBinaryProto(tensorflow::Env::Default(),
			fileProp.filepath,
			&graph_def);
	if (!load_graph_status.ok()) {
		cout << "[LoadGraph] load graph failed!\n";
		return;
	}

	tensorflow::SessionOptions options;
	//tensorflow::ConfigProto* config = &options.config;
	options.config.mutable_device_count()->insert({ "GPU",1 });
	options.config.mutable_gpu_options()->set_allow_growth(true);
	options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
	m_session.reset(tensorflow::NewSession(options));
	auto status_creat_session = m_session.get()->Create(graph_def);
	std::cout << "create session success\n";
	if (!status_creat_session.ok()) {
		std::cout << "[LoadGraph] creat session failed!\n" << std::endl;
		return;
	}
}

void TfBase::output(tensorflow::Tensor& tensorInput, vector<tensorflow::Tensor>& tensorOutput)
{
	auto status_run = m_session->Run({ { fileProp.inputName,tensorInput } },
		fileProp.outputNames, {}, &tensorOutput);
	if (!status_run.ok()) {
		std::cout << "run model failed!\n";
	}
}

void TfBase::Mat2Tensor(std::vector<cv::Mat>& imgs, tensorflow::Tensor& dstTensor)
{
	int size = imgs.size();
	if (size == 0)
		return;
	int height = imgs[0].rows;
	int width = imgs[0].cols;
	int channel = imgs[0].channels();
	for (int i = 0; i < size; i++)
	{
		float* ptr = dstTensor.flat<float>().data() + i * height * width * channel;
		cv::Mat tensor_image(height, width, CV_32FC3, ptr);
		imgs[i].convertTo(tensor_image, CV_32F);//转为float类型的数组
		tensor_image = (tensor_image / 255 - 0.5) * 2;
	}
}

void TfBase::output(std::vector<cv::Mat>& imgs, std::vector<tensorflow::Tensor>& Output)
{
	int size = imgs.size();
	if (size == 0)
		return;
	int height = imgs[0].rows;
	int width = imgs[0].cols;
	int channel = imgs[0].channels();
	tensorflow::Tensor tem_tensor_res(tensorflow::DataType::DT_FLOAT,
		tensorflow::TensorShape({ size, height, width, channel }));
	Mat2Tensor(imgs, tem_tensor_res);
	output(tem_tensor_res, Output);
}

void TfBase::convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs)
{
	//1.先将imgs转为Tensor
	int size = imgs.size();
	if (size == 0)
		return;
	int height = imgs[0].rows;
	int width = imgs[0].cols;
	int channel = imgs[0].channels();
	tensorflow::Tensor dstTensor(tensorflow::DataType::DT_FLOAT,
		tensorflow::TensorShape({ size, height, width, channel }));
	Mat2Tensor(imgs, dstTensor);
	//2.取得锁
	std::unique_lock<std::mutex> myGuard(queue_lock);
	//3.将dstTensor放到队列里面
	tensorQueue.emplace(std::move(dstTensor));
	//4.解锁
	myGuard.unlock();
	//5.通过条件变量通知另一个等待线程：队列里有数据了！
	tensor_queue_cv.notify_one();
}

bool TfBase::checkQueueEmpty()
{
	if (tensorQueue.empty())
		return true;
	else
		return false;
}

