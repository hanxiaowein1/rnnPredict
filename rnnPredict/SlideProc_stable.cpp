#include "SlideProc.h"
#include "tinyxml.h"
#include "tinystr.h"
#include "IniConfig.h"
void SlideProc::model1Config()
{
	model1Mpp = IniConfig::instance().getIniDouble("Model1", "mpp");
	model1Height = IniConfig::instance().getIniInt("Model1", "height");
	model1Width = IniConfig::instance().getIniInt("Model1", "width");
	//model1Mpp = 0.586f;
	//model1Height = 512;
	//model1Width = 512;
}

void SlideProc::model2Config()
{
	//model2Mpp = 0.293f;
	//model2Height = 256;
	//model2Width = 256;
	model2Mpp = IniConfig::instance().getIniDouble("Model2", "mpp");
	model2Height = IniConfig::instance().getIniInt("Model2", "height");
	model2Width = IniConfig::instance().getIniInt("Model2", "width");
}

void SlideProc::initialize_handler(const char* iniPath)
{
	string modelConfigIni = string(iniPath);

	model1Config();
	m1Holder = std::make_unique<Model1Holder>(modelConfigIni);
	//m1Holder->createThreadPool(3);
	model2Config();
	m2Holder = std::make_unique<Model2Holder>(modelConfigIni);
	m2Holder->createThreadPool(2);
	//model3Config(string(model3Path));
	m3Holder = std::make_unique<Model3Holder>(modelConfigIni);
	m3Holder->createThreadPool(1);
	rnnHolder = std::make_unique<RnnHolder>(modelConfigIni);
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
