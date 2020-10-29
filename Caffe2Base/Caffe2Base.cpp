#include "Caffe2Base.h"
#include "cv_extend.h"
#include "CommonFunction.h"

Caffe2Base::Caffe2Base(std::string group)
{
	fileProp.initByIniConfig(group);
	construct();
}

void Caffe2Base::construct()
{
	caffe2::GlobalInit();
	std::vector<std::string> two_nets = split(fileProp.filepath, ',');
	//第一个是initnet，第二个是predict_net
	std::string init_net_path = two_nets[0];
	std::string predict_net_path = two_nets[1];

	caffe2::DeviceOption option;
	option.set_device_type((int)caffe2::CUDA);
	new caffe2::CUDAContext(option);

	// initialize Net and Workspace
	//caffe2::NetDef initNet_, predictNet_;
	predictNet_.mutable_device_option()->set_device_id(0);
	initNet_.mutable_device_option()->set_device_id(0);
	predictNet_.mutable_device_option()->set_device_type((int)caffe2::CUDA);
	initNet_.mutable_device_option()->set_device_type((int)caffe2::CUDA);
	CAFFE_ENFORCE(ReadProtoFromFile(init_net_path, &initNet_));
	CAFFE_ENFORCE(ReadProtoFromFile(predict_net_path, &predictNet_));

	for (auto& str : predictNet_.external_input()) {
		work_space.CreateBlob(str);
	}
	CAFFE_ENFORCE(work_space.CreateNet(predictNet_, true));
	CAFFE_ENFORCE(work_space.RunNetOnce(initNet_));
	
}

void Caffe2Base::convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs)
{
	int size = imgs.size();
	if (size == 0)
		return;
	resizeImages(imgs, inputProp.height, inputProp.width);
	int height = imgs[0].rows;
	int width = imgs[0].cols;
	int channel = imgs[0].channels();
	std::vector<float> neededData(height * width * channel * size);
	transformInMemory(imgs, neededData.data());
	//将其塞到队列里
	std::unique_lock<std::mutex> myGuard(queue_lock);
	std::pair<int, std::vector<float>> temp_elem;
	temp_elem.first = imgs.size();
	temp_elem.second = std::move(neededData);
	tensorQueue.emplace(std::move(temp_elem));
	myGuard.unlock();
	//通过条件变量通知另一个等待线程：队列里有数据了！
	tensor_queue_cv.notify_one();
}

bool Caffe2Base::checkQueueEmpty()
{
	if (tensorQueue.empty())
		return true;
	else
		return false;
}

void Caffe2Base::runNet(std::vector<float>& input_data)
{
	std::vector<std::string> input_names = split(fileProp.inputName, ',');//第一个是data，第二个是info
	std::string data_name = input_names[0];//这里都不太好，用逗号隔离，容易写反，以后换成json格式的，反过来也就无所谓了
	std::string info_name = input_names[1];

	caffe2::TensorCUDA* data = BlobGetMutableTensor(work_space.GetBlob(data_name), caffe2::CUDA);
	int tensorBatch = input_data.size() /
		(inputProp.height * inputProp.width * inputProp.channel);
	caffe2::TensorCPU input_tensor(
		caffe2::IntArrayRef{ tensorBatch , inputProp.channel, inputProp.height , inputProp.width },
		caffe2::DeviceType::CPU);
	input_tensor.ShareExternalPointer((float*)input_data.data());
	data->CopyFrom(input_tensor);

	auto im_info = BlobGetMutableTensor(work_space.GetBlob(info_name), caffe2::CUDA);
	caffe2::TensorCPU inputTensor2(caffe2::IntArrayRef{ tensorBatch, inputProp.channel }, caffe2::DeviceType::CPU);
	std::vector<float> vecData2{ (float)inputProp.height, (float)inputProp.width, 1.0 };
	inputTensor2.ShareExternalPointer(vecData2.data());
	im_info->CopyFrom(inputTensor2);

	std::string predictNetName = predictNet_.name();
	CAFFE_ENFORCE(work_space.RunNet(predictNet_.name()));

}