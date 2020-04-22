#include "model.h"

model::model()
{

}

model::model(modelConfig config, char* buffer, int size)
{
	if (config.width == 0)
	{
		std::cout << "width cannot be zero!\n";
		return;
	}
	m_width = config.width;
	if (config.height == 0)
	{
		std::cout << "height cannot be zero!\n";
		return;
	}
	m_height = config.height;
	if (config.channel == 0)
	{
		std::cout << "channel cannot be zero!\n";
		return;
	}
	m_channel = config.channel;
	if (config.opsInput == "")
	{
		std::cout << "opsInput cannot be null str!\n";
		return;
	}
	m_opsInput = config.opsInput;
	if (config.opsOutput.size() == 0)
	{
		std::cout << "opsOutput size cannot be zero!\n";
		return;
	}
	m_opsOutput = config.opsOutput;

	//配置tensorflow的session
	tensorflow::GraphDef graph_def;
	if (!graph_def.ParseFromArray(buffer, size))
	{
		std::cout << "load graph failed!\n";
		return;
	}
	SessionOptions options;
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

void model::output(Tensor &tensorInput, vector<Tensor> &tensorOutput)
{
	auto status_run = m_session->Run({ { m_opsInput,tensorInput } },
		m_opsOutput, {}, &tensorOutput);
	if (!status_run.ok()) {
		std::cout << "run model failed!\n";
	}
}

void model::output(std::vector<cv::Mat>& imgs, std::vector<Tensor>& Output, int flag)
{
	int batchsize = imgs.size();
	//tensorflow::Tensor tem_tensor_res(tensorflow::DataType::DT_UINT8, tensorflow::TensorShape({ batchsize, m_height, m_width, m_channel }));
	tensorflow::Tensor tem_tensor_res(tensorflow::DataType::DT_FLOAT, 
		tensorflow::TensorShape({ batchsize, imgs[0].rows, imgs[0].cols, m_channel }));
	auto mapTensor = tem_tensor_res.tensor<float, 4>();
	for (int batch = 0; batch < batchsize; batch++) {
		for (int h = 0; h < imgs[0].rows; h++) {
			auto linePtr = imgs[batch].ptr(h);
			for (int w = 0; w < imgs[0].cols; w++) {
				mapTensor(batch, h, w, 0) = *(linePtr + w * 3);
				mapTensor(batch, h, w, 1) = *(linePtr + w * 3 + 1);
				mapTensor(batch, h, w, 2) = *(linePtr + w * 3 + 2);
			}
		}
	}

	auto status_run = m_session->Run({ { m_opsInput,tem_tensor_res } },
		m_opsOutput, {}, &Output);
	if (!status_run.ok()) {
		std::cout << "run model failed!\n";
	}

}

void model::output(std::vector<cv::Mat> &imgs, std::vector<Tensor> &Output)
{
	int batchsize = imgs.size();
	tensorflow::Tensor tem_tensor_res(tensorflow::DataType::DT_FLOAT, 
		tensorflow::TensorShape({ batchsize, m_height, m_width, m_channel }));
	for (int i = 0; i < batchsize; i++)
	{
		float* ptr = tem_tensor_res.flat<float>().data() + i * m_height * m_width * m_channel;
		cv::Mat tensor_image(m_height, m_width, CV_32FC3, ptr);
		imgs[i].convertTo(tensor_image, CV_32F);//转为float类型的数组
		tensor_image = (tensor_image / 255 - 0.5) * 2;
	}
	auto status_run = m_session->Run({ { m_opsInput,tem_tensor_res } },
		m_opsOutput, {}, &Output);
	if (!status_run.ok()) {
		std::cout << "run model failed!\n";
	}

	//int batchsize = imgs.size();
	//tensorflow::Tensor tem_tensor_res(tensorflow::DataType::DT_FLOAT,
	//	tensorflow::TensorShape({ batchsize, m_height, m_width, m_channel }));
	//auto mapTensor = tem_tensor_res.tensor<float, 4>();

	//for (int batch = 0; batch < batchsize; batch++)
	//{
	//	for (int h = 0; h < m_height; h++)
	//	{
	//		auto linePtr = imgs[batch].ptr(h);
	//		for (int w = 0; w < m_width; w++)
	//		{
	//			mapTensor(batch, h, w, 0) = (((*(linePtr + w * 3))*1.0f) / 255 - 0.5) * 2;
	//			mapTensor(batch, h, w, 1) = (((*(linePtr + w * 3 + 1))*1.0f) / 255 - 0.5) * 2;
	//			mapTensor(batch, h, w, 2) = (((*(linePtr + w * 3 + 2))*1.0f) / 255 - 0.5) * 2;
	//		}
	//	}
	//}
	////tensorInput.CopyFrom(tem_tensor_res, tensorflow::TensorShape({ batchsize, m_height, m_width, m_channel }));
	//auto status_run = m_session->Run({ { m_opsInput,tem_tensor_res } },
	//	m_opsOutput, {}, &Output);
	//if (!status_run.ok()) {
	//	std::cout << "run model failed!\n";
	//}
}

