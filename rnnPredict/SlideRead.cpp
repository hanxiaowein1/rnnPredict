#include "SlideRead.h"



SlideRead::SlideRead()
{
	
}


SlideRead::~SlideRead()
{
	if (buffer != nullptr)
		delete[]buffer;
}

void SlideRead::ini_ration()
{
	int level1Height = 0;
	int level1Width = 0;
	int level0Height = 0;
	int level0Width = 0;
	getLevelDimensions(0, level0Width, level0Height);
	getLevelDimensions(1, level1Width, level1Height);
	if (level0Height == 0 || level0Width == 0 || level1Height == 0 || level1Width == 0)
	{
		return;
	}
	double level0HeightDouble = level0Height;
	double level1HeightDouble = level1Height;
	m_ratio = std::round(double(level0HeightDouble / level1HeightDouble));
}

void SlideRead::bufferManage(int width, int height, int channel)
{
	if (m_width == 0)
		m_width = width;
	if (m_height == 0)
		m_height = height;
	if (m_channel == 0)
		m_channel = channel;
	if (buffer == nullptr)
		buffer = new unsigned char[m_width * m_height * channel];
	if (m_width != width || m_height != height || m_channel != channel)
	{
		//œ»…æ≥˝‘⁄∑÷≈‰
		delete[]buffer;
		m_width = width;
		m_height = height;
		m_channel = channel;
		buffer = new unsigned char[m_width * m_height * m_channel];
	}
}
