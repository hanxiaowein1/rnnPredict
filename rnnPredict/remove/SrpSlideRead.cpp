#include "SrpSlideRead.h"



SrpSlideRead::SrpSlideRead()
{
}

SrpSlideRead::SrpSlideRead(const char * slidePath):SlideRead()
{
	iniFunction();
	iniHandle(slidePath);
	ini_ration();
}


SrpSlideRead::~SrpSlideRead()
{
	Close(m_srpHandle);
	//由外部类进行释放，不由内部进行反复加载释放
	//if (srpDll != nullptr)
	//{
	//	FreeLibrary(srpDll);
	//}
}

void SrpSlideRead::iniFunction()
{
	srpDll = LoadLibraryA("srp.dll");
	if (srpDll != nullptr)
	{
		Open = (OpenRW_function)GetProcAddress(srpDll, "OpenRW");
		if (Open == nullptr)
			std::cout << "Open is null\n";
		ReadParamInt32 = (ReadParamInt32_function)GetProcAddress(srpDll, "ReadParamInt32");
		if (ReadParamInt32 == nullptr)
			std::cout << "ReadParamInt32 is null\n";
		ReadParamDouble = (ReadParamDouble_function)GetProcAddress(srpDll, "ReadParamDouble");
		if (ReadParamDouble == nullptr)
			std::cout << "ReadParamDouble is null\n";
		WriteParamDouble = (WriteParamDouble_function)GetProcAddress(srpDll, "WriteParamDouble");
		if (WriteParamDouble == nullptr)
			cout << "WriteParamDouble is null\n";
		ReadRegionRGB = (ReadRegionRGB_function)GetProcAddress(srpDll, "ReadRegionRGB");
		if (ReadRegionRGB == nullptr)
			std::cout << "ReadRegionRGB is null\n";
		CleanAnno = (CleanAnno_function)GetProcAddress(srpDll, "CleanAnno");
		if (CleanAnno == nullptr)
			cout << "CleanAnno is null\n";
		BeginBatch = (BeginBatch_function)GetProcAddress(srpDll, "BeginBatch");
		if (BeginBatch == nullptr)
			cout << "BeginBatch is null\n";
		WriteAnno = (WriteAnno_function)GetProcAddress(srpDll, "WriteAnno");
		if (WriteAnno == nullptr)
			cout << "WriteAnno is null\n";
		EndBatch = (EndBatch_function)GetProcAddress(srpDll, "EndBatch");
		if (EndBatch == nullptr)
			cout << "EndBatch is null\n";
		Close = (Close_function)GetProcAddress(srpDll, "Close");
		if (Close == nullptr)
			std::cout << "Close is null\n";
	}
	else
	{
		std::cout << "srpDll is null\n";
	}
}

void SrpSlideRead::iniHandle(const char* slidePath)
{
	m_srpHandle = Open(slidePath);
}

void SrpSlideRead::getSlideWidth(int & width)
{
	ReadParamInt32(m_srpHandle, "width", &width);
}

void SrpSlideRead::getSlideHeight(int & height)
{
	ReadParamInt32(m_srpHandle, "height", &height);
}

void SrpSlideRead::getSlideBoundX(int & boundX)
{
	boundX = 0;
}

void SrpSlideRead::getSlideBoundY(int & boundY)
{
	boundY = 0;
}

void SrpSlideRead::getSlideMpp(double & mpp)
{
	ReadParamDouble(m_srpHandle, "mpp", &mpp);
}

void SrpSlideRead::getLevelDimensions(int level, int & width, int & height)
{
	std::string wStr = "level_widths[" + std::to_string(level) + "]";
	std::string hStr = "level_heights[" + std::to_string(level) + "]";
	ReadParamInt32(m_srpHandle, wStr.c_str(), &width);
	ReadParamInt32(m_srpHandle, hStr.c_str(), &height);
}

void SrpSlideRead::callCleanAnno()
{
	CleanAnno(m_srpHandle);
}

bool SrpSlideRead::callBeginBatch()
{
	return BeginBatch(m_srpHandle);
}

void SrpSlideRead::callWriteAnno(Anno* anno, int count)
{
	WriteAnno(m_srpHandle, anno, count);
}

bool SrpSlideRead::callEndBatch()
{
	return EndBatch(m_srpHandle);
}

void SrpSlideRead::callWriteParamDouble(const char* key, double value)
{
	WriteParamDouble(m_srpHandle, key, value);
}

bool SrpSlideRead::status()
{
	if(m_srpHandle == 0)
	    return false;
	return true;
}

void SrpSlideRead::getTile(int level, int x, int y, int width, int height, cv::Mat &img)
{
	//std::unique_ptr<unsigned char[]> uBuffer(new unsigned char[width * height * 3]);
	bufferManage(width, height, 3);
	int len = 0;
	ReadRegionRGB(m_srpHandle, level, x, y, width, height, buffer, &len);
	img = cv::Mat(height, width, CV_8UC3, buffer, cv::Mat::AUTO_STEP).clone();
}
