#include "DetectModel.h"

DetectModel::DetectModel(std::string group) : Caffe2Base(group)
{
	inputProp.initByIniConfig("DetectModel");
}

void DetectModel::clearResult()
{
	m_result.clear();
}

void DetectModel::processFirstDataInQueue()
{
	//处理队列的第一个元素
	std::vector<float> input_data = std::move(tensorQueue.front());
	tensorQueue.pop();
	int tensorBatch = input_data.size() /
		(inputProp.height * inputProp.width * inputProp.channel);
	runNet(input_data);

	//处理workspace中的数据(暂时就直接返回debugString)

	for (int i = 0; i < fileProp.outputNames.size(); i++)
	{
		auto &result = caffe2::BlobGetTensor(*work_space.GetBlob(fileProp.outputNames[i]), caffe2::CUDA);
		//m_results.emplace_back(result.DebugString());
		m_result[fileProp.outputNames[i]] = result.DebugString();
	}
}