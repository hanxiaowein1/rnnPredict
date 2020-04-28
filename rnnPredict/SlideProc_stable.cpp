#include "SlideProc.h"
#include "tinyxml.h"
#include "tinystr.h"

void SlideProc::rnnConfig(string rnnParentPath)
{
	rnnHolder = new RnnHolder(rnnParentPath);
}

void SlideProc::model1Config(string model1Path)
{
	model1Mpp = 0.586f;
	model1Height = 512;
	model1Width = 512;
}

void SlideProc::model2Config(string model2Path)
{
	model2Mpp = 0.293f;
	model2Height = 256;
	model2Width = 256;
}

void SlideProc::initialize_handler(const char* iniPath)
{
	//读取ini文件中的相关信息
	char model1Path[MAX_PATH];
	char model2Path[MAX_PATH];
	char model3Path[MAX_PATH];
	char rnnParentPath[MAX_PATH];
	//char xgParentPath[MAX_PATH];
	char group[] = "Config";
	char pbFile_n_1[] = "model1Path";
	char pbFile_n_2[] = "model2Path";
	char pbFile_n_3[] = "model3Path";
	char pbFile_n_6[] = "rnnParentPath";
	//char pbFile_n_7[] = "xgParentPath";

	GetPrivateProfileString(group, pbFile_n_1, "default", model1Path, MAX_PATH, iniPath);
	GetPrivateProfileString(group, pbFile_n_2, "default", model2Path, MAX_PATH, iniPath);
	GetPrivateProfileString(group, pbFile_n_3, "default", model3Path, MAX_PATH, iniPath);
	GetPrivateProfileString(group, pbFile_n_6, "default", rnnParentPath, MAX_PATH, iniPath);
	//GetPrivateProfileString(group, pbFile_n_7, "default", xgParentPath, MAX_PATH, iniPath);

	string modelConfigIni = "../x64/Release/ModelConfig.ini";

	model1Config(string(model1Path));
	m1Holder = new Model1Holder(modelConfigIni);
	m1Holder->createThreadPool(3);
	model2Config(string(model2Path));
	m2Holder = new Model2Holder(modelConfigIni);
	m2Holder->createThreadPool(2);
	//model3Config(string(model3Path));
	m3Holder = new Model3Holder(modelConfigIni);
	m3Holder->createThreadPool(1);
	rnnConfig(string(modelConfigIni));
	//xgConfig(string(xgParentPath));
}

void SlideProc::saveResult(string savePath, string filename)
{
	TiXmlDocument* writeDoc = new TiXmlDocument;//xml文档指针	
	TiXmlDeclaration* decl = new TiXmlDeclaration("1.0", "UTF-8", "yes");//文档格式声明
	writeDoc->LinkEndChild(decl);//写入文档

	TiXmlElement* RootElement = new TiXmlElement("Slide");
	RootElement->SetAttribute("name", filename.c_str());
	RootElement->SetAttribute("num", rResults.size());
	writeDoc->LinkEndChild(RootElement);

	//int count = 0;
	for (auto& elem : rResults) {
		TiXmlElement* BlockElem = new TiXmlElement("Block");
		RootElement->LinkEndChild(BlockElem);
		BlockElem->SetAttribute("X", elem.point.x);
		BlockElem->SetAttribute("Y", elem.point.y);
		BlockElem->SetAttribute("score", to_string(elem.result.score).c_str());
		bool model2Flag = false;
		if (elem.score2.size() > 0)
			model2Flag = true;
		for (auto iter = elem.result.points.begin() + 1; iter != elem.result.points.end(); iter++) {
			int place = iter - elem.result.points.begin() - 1;
			TiXmlElement* LocateElem = new TiXmlElement("Locate");
			BlockElem->LinkEndChild(LocateElem);
			if (model2Flag) {
				LocateElem->SetAttribute("score", to_string(elem.score2[place]).c_str());
			}
			else {
				LocateElem->SetAttribute("score", "-1");
			}
			cv::Point point;
			point.x = elem.point.x + iter->x * float(model1Mpp / slideMpp);
			point.y = elem.point.y + iter->y * float(model1Mpp / slideMpp);
			//cv::Mat img;
			int saveHeight = 256 * float(model2Mpp / slideMpp);
			//m_sdpcRead->getTile(0, point.x - saveHeight / 2, point.y - saveHeight / 2, saveHeight, saveHeight, img);
			//cv::imwrite("D:\\TEST_OUTPUT\\rnnPredict\\" + to_string(count) + ".tif", img);
			//count++;
			LocateElem->SetAttribute("X", point.x);
			LocateElem->SetAttribute("Y", point.y);
		}
	}
	writeDoc->SaveFile((savePath + "\\" + filename + ".xml").c_str());
	delete writeDoc;
}
