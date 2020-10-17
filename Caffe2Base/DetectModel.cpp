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
	//������еĵ�һ��Ԫ��
	std::vector<float> input_data = std::move(tensorQueue.front());
	tensorQueue.pop();
	int tensorBatch = input_data.size() /
		(inputProp.height * inputProp.width * inputProp.channel);
	runNet(input_data);

	//����workspace�е�����(��ʱ��ֱ�ӷ���debugString)

	for (int i = 0; i < fileProp.outputNames.size(); i++)
	{
		auto &result = caffe2::BlobGetTensor(*work_space.GetBlob(fileProp.outputNames[i]), caffe2::CUDA);
		//m_results.emplace_back(result.DebugString());
		m_result[fileProp.outputNames[i]] = result.DebugString();
	}
}